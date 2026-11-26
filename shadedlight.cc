/*
 *  shadedlight.cc - Height-aware "shaded light" scene record.
 *
 *  Copyright (C) 2025  The Exult Team
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 */

#ifdef HAVE_CONFIG_H
#	include <config.h>
#endif

#include "shadedlight.h"

#include "Configuration.h"
#include "actors.h"
#include "chunks.h"
#include "data/exult_bg_flx.h"
#include "data/exult_si_flx.h"
#include "data_utils.h"
#include "exceptions.h"
#include "exult.h"
#include "game.h"
#include "gameclk.h"
#include "gamemap.h"
#include "gamewin.h"
#include "ibuf8.h"
#include "ignore_unused_variable_warning.h"
#include "iwin8.h"
#include "palette.h"
#include "ready.h"
#include "shapeinf.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

#ifdef HAVE_PNG_H
#	include <png.h>
#endif

// The classifier, on a 1x1 tile, 4-lift box (8 x 8 px footprint, 16 px high).
static_assert(Shaded_classify_box(-20, -20, 8, 8, 16).facing == shaded_face_top);
static_assert(Shaded_classify_box(-20, -20, 8, 8, 16).dz_px == 16);
static_assert(Shaded_classify_box(0, -5, 8, 8, 16).facing == shaded_face_east);
static_assert(Shaded_classify_box(0, -5, 8, 8, 16).dz_px == 0);
static_assert(Shaded_classify_box(-3, 0, 8, 8, 16).facing == shaded_face_south);
static_assert(Shaded_classify_box(-3, 0, 8, 8, 16).dz_px == 0);
static_assert(Shaded_classify_box(-10, -6, 8, 8, 16).facing == shaded_face_south);
static_assert(Shaded_classify_box(-10, -6, 8, 8, 16).dz_px == 6);
static_assert(Shaded_classify_box(-40, -40, 8, 8, 16).facing == (shaded_face_east | shaded_face_clamped));
static_assert(Shaded_classify_box(-40, -40, 8, 8, 16).dz_px == 16);
// Flats are ground at the bottom of the box.
static_assert(Shaded_classify_box(-3, -3, 8, 8, 0).facing == shaded_face_ground);
static_assert(Shaded_classify_box(-3, -9, 8, 8, 0).facing == (shaded_face_ground | shaded_face_clamped));
// Mirrored frames swap x and y sizes, which swaps the south and east faces.
static_assert(Shaded_classify_box(-10, -6, 8, 16, 16).facing == shaded_face_south);
static_assert(Shaded_classify_box(-6, -10, 16, 8, 16).facing == shaded_face_east);
static_assert(Shaded_classify_box(-10, -6, 8, 16, 16).dz_px == Shaded_classify_box(-6, -10, 16, 8, 16).dz_px);
// The default rows: candle, torch and lamp.
static_assert(Shaded_default_light(1).radius == 4 && Shaded_default_light(1).brightness == 60);
static_assert(Shaded_default_light(5).radius == 8 && Shaded_default_light(5).brightness == 100);
static_assert(Shaded_default_light(7).radius == 10 && Shaded_default_light(7).brightness == 120);
// Falloff: full at the light, a quarter half way out (in d^2), none past R.
static_assert(Shaded_falloff(0.0f) == 1.0f);
static_assert(Shaded_falloff(0.5f) == 0.25f);
static_assert(Shaded_falloff(1.0f) == 0.0f && Shaded_falloff(2.0f) == 0.0f);
// Dither thresholds: inside (0, 1), repeating every 4 pixels.
static_assert(Shaded_bayer4(0, 0) == 0.5f / 16.0f && Shaded_bayer4(3, 3) == 5.5f / 16.0f);
static_assert(Shaded_bayer4(0, 3) == 15.5f / 16.0f && Shaded_bayer4(4, -4) == Shaded_bayer4(0, 0));
// Facing: a south face sees lights to its south, an east face lights to
// its east, and tops and ground only lights above them.
static_assert(Shaded_faces_light(shaded_face_south, 0.0f, 2.0f, 0.0f));
static_assert(!Shaded_faces_light(shaded_face_south, 0.0f, -2.0f, 0.0f));
static_assert(Shaded_faces_light(shaded_face_east, 2.0f, 0.0f, 0.0f));
static_assert(!Shaded_faces_light(shaded_face_east, -2.0f, 0.0f, 0.0f));
static_assert(Shaded_faces_light(shaded_face_top, 1.0f, 1.0f, 0.5f));
static_assert(!Shaded_faces_light(shaded_face_top, 1.0f, 1.0f, -0.5f));
static_assert(!Shaded_faces_light(shaded_face_ground | shaded_face_clamped, 0.0f, 0.0f, -1.0f));

/*
 *  A small persistent thread pool for the light pass.  run() hands bands
 *  out to the workers and the calling thread, and returns when all are
 *  done.
 */
class Shaded_workers {
public:
	explicit Shaded_workers(int threads) {
		for (int i = 1; i < threads; i++) {
			workers.emplace_back([this]() {
				work();
			});
		}
	}

	~Shaded_workers() {
		{
			const std::lock_guard<std::mutex> lock(mutex);
			stop = true;
		}
		wake.notify_all();
		for (auto& worker : workers) {
			worker.join();
		}
	}

	Shaded_workers(const Shaded_workers&)            = delete;
	Shaded_workers& operator=(const Shaded_workers&) = delete;

	int size() const {
		return static_cast<int>(workers.size()) + 1;
	}

	void run(int bands, const std::function<void(int)>& job) {
		{
			const std::lock_guard<std::mutex> lock(mutex);
			current    = &job;
			band_count = bands;
			next_band  = 0;
			remaining  = bands;
			generation++;
		}
		wake.notify_all();
		std::unique_lock<std::mutex> lock(mutex);
		take_bands(lock);
		done.wait(lock, [this]() {
			return remaining == 0;
		});
		current = nullptr;
	}

private:
	std::vector<std::thread>          workers;
	std::mutex                        mutex;
	std::condition_variable           wake;
	std::condition_variable           done;
	const std::function<void(int)>*   current    = nullptr;
	int                               band_count = 0;
	int                               next_band  = 0;
	int                               remaining  = 0;
	uint64_t                          generation = 0;
	bool                              stop       = false;

	// Run bands until none are left; the lock is held on entry and exit.
	void take_bands(std::unique_lock<std::mutex>& lock) {
		while (next_band < band_count) {
			const int                       band = next_band++;
			const std::function<void(int)>* job  = current;
			lock.unlock();
			(*job)(band);
			lock.lock();
			if (--remaining == 0) {
				done.notify_all();
			}
		}
	}

	void work() {
		uint64_t                     seen = 0;
		std::unique_lock<std::mutex> lock(mutex);
		while (true) {
			wake.wait(lock, [this, seen]() {
				return stop || generation != seen;
			});
			if (stop) {
				return;
			}
			seen = generation;
			take_bands(lock);
		}
	}
};

namespace {
	constexpr int lut_size = 64;    // False colour codes per view.

	struct Rgb {
		int r, g, b;
	};

	Shaded_rect Intersect(const Shaded_rect& a, const Shaded_rect& b) {
		const int x0 = std::max(a.x, b.x);
		const int y0 = std::max(a.y, b.y);
		const int x1 = std::min(a.x + a.w, b.x + b.w);
		const int y1 = std::min(a.y + a.h, b.y + b.h);
		if (x1 <= x0 || y1 <= y0) {
			return Shaded_rect{};
		}
		return Shaded_rect{x0, y0, x1 - x0, y1 - y0};
	}

	// Blue (low) through cyan, green and yellow to red (high); t in [0, 1].
	Rgb Height_colour(float t) {
		static const Rgb stops[] = {
				{30, 40, 150},
				{40, 170, 220},
				{60, 200, 80},
				{240, 220, 60},
				{230, 50, 40}
		};
		t               = std::clamp(t, 0.0f, 1.0f) * 4.0f;
		const int   i   = std::min(static_cast<int>(t), 3);
		const float f   = t - static_cast<float>(i);
		auto        mix = [f](int a, int b) {
			return static_cast<int>(static_cast<float>(a) + (static_cast<float>(b - a) * f));
		};
		return Rgb{mix(stops[i].r, stops[i + 1].r), mix(stops[i].g, stops[i + 1].g), mix(stops[i].b, stops[i + 1].b)};
	}

	Rgb Facing_colour(uint8_t facing) {
		Rgb col;
		switch (facing & shaded_face_mask) {
		case shaded_face_ground:
			col = Rgb{60, 140, 60};
			break;
		case shaded_face_top:
			col = Rgb{235, 215, 80};
			break;
		case shaded_face_south:
			col = Rgb{70, 110, 235};
			break;
		case shaded_face_east:
			col = Rgb{225, 70, 60};
			break;
		case shaded_face_rounded:
			col = Rgb{200, 120, 220};
			break;
		default:
			return Rgb{0, 0, 0};
		}
		if (facing & shaded_face_clamped) {
			col = Rgb{col.r * 6 / 10, col.g * 6 / 10, col.b * 6 / 10};
		}
		return col;
	}

	// Well spread, fairly bright colour per object id; black for none.
	Rgb Id_colour(unsigned id) {
		if (id == shaded_id_none) {
			return Rgb{0, 0, 0};
		}
		const uint32_t h = static_cast<uint32_t>(id) * 2654435761u;
		return Rgb{static_cast<int>(64 + ((h >> 24) % 192)), static_cast<int>(64 + ((h >> 16) % 192)),
				   static_cast<int>(64 + ((h >> 8) % 192))};
	}

	// Face and height of one pixel of an object; (u, v) from its origin.
	inline Shaded_box_hit Classify(const Shaded_object_context& obj, int u, int v) {
		switch (obj.volume) {
		case Shaded_volume::rounded:
			// Later stage: a rounded volume inside the box.  Box for now.
		case Shaded_volume::box:
			break;
		}
		return Shaded_classify_box(u, v, obj.xpx, obj.ypx, obj.hpx);
	}

	// Raw row pointers into the three buffers (fast even unoptimised).
	struct Shaded_row {
		uint16_t* height;
		uint8_t*  facing;
		uint16_t* ids;

		Shaded_row(Shaded_scene_buffers& buf, int x, int y) {
			const int idx = buf.index(x, y);
			height        = buf.height_map.data() + idx;
			facing        = buf.facing_map.data() + idx;
			ids           = buf.object_ids.data() + idx;
		}
	};

#ifdef HAVE_PNG_H
	// Write rows of 8-bit RGB (or 16-bit big-endian grey) as a PNG.
	bool Write_png(const std::string& path, int w, int h, bool grey16, const std::vector<unsigned char>& data) {
		FILE* fp = fopen(path.c_str(), "wb");
		if (!fp) {
			return false;
		}
		png_structp png  = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
		png_infop   info = png ? png_create_info_struct(png) : nullptr;
		if (!png || !info) {
			png_destroy_write_struct(&png, &info);
			fclose(fp);
			return false;
		}
		if (setjmp(png_jmpbuf(png))) {
			png_destroy_write_struct(&png, &info);
			fclose(fp);
			return false;
		}
		png_init_io(png, fp);
		png_set_IHDR(
				png, info, static_cast<png_uint_32>(w), static_cast<png_uint_32>(h), grey16 ? 16 : 8,
				grey16 ? PNG_COLOR_TYPE_GRAY : PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
				PNG_FILTER_TYPE_DEFAULT);
		png_write_info(png, info);
		const size_t                row_bytes = static_cast<size_t>(w) * (grey16 ? 2 : 3);
		std::vector<png_const_bytep> rows(static_cast<size_t>(h));
		for (int y = 0; y < h; y++) {
			rows[static_cast<size_t>(y)] = data.data() + static_cast<size_t>(y) * row_bytes;
		}
		png_write_rows(png, const_cast<png_bytepp>(rows.data()), static_cast<png_uint_32>(h));
		png_write_end(png, info);
		png_destroy_write_struct(&png, &info);
		fclose(fp);
		return true;
	}
#endif

	using Light_table = std::map<std::pair<int, int>, Shaded_light_entry>;

	/*
	 *  Reads the lights section of shaded_light.txt:
	 *      shape/frame/radius/brightness/red/green/blue/height/flags
	 *  Patch rows replace static rows with the same shape and frame.
	 */
	class Shaded_light_reader final : public Base_reader {
		Light_table& table;

	protected:
		void read_data(std::istream& in, size_t index, int version, bool patch, Exult_Game game, bool binary) override {
			ignore_unused_variable_warning(index, version, patch, game, binary);
			const int shape = ReadInt(in, -1);
			int       frame = ReadInt(in, -1);
			if (shape < 0) {
				return;
			}
			frame = frame < 0 ? -1 : (frame & 0xff);
			// Missing fields take a torch's row.
			const Shaded_light_entry def = Shaded_default_light(5);
			Shaded_light_entry       entry;
			entry.radius     = std::max(ReadInt(in, def.radius), 0);
			entry.brightness = std::max(ReadInt(in, def.brightness), 0);
			entry.red        = std::clamp(ReadInt(in, def.red), 0, 255);
			entry.green      = std::clamp(ReadInt(in, def.green), 0, 255);
			entry.blue       = std::clamp(ReadInt(in, def.blue), 0, 255);
			entry.height     = ReadInt(in, def.height);
			entry.flags      = ReadInt(in, def.flags) & (shaded_light_area | shaded_light_off);

			table[std::make_pair(shape, frame)] = entry;
		}

	public:
		explicit Shaded_light_reader(Light_table& t) : Base_reader(false), table(t) {}
	};

	// FNV-1a over 32-bit values.
	void Hash_add(uint64_t& hash, int32_t value) {
		auto v = static_cast<uint32_t>(value);
		for (int i = 0; i < 4; i++) {
			hash ^= v & 0xffu;
			hash *= 1099511628211ull;
			v >>= 8;
		}
	}

	const char* Row_name(Shaded_light_row row) {
		switch (row) {
		case Shaded_light_row::exact:
			return "table exact";
		case Shaded_light_row::wildcard:
			return "table wildcard";
		case Shaded_light_row::formula:
			break;
		}
		return "default";
	}
}    // namespace

Shaded_rect Shaded_scene_buffers::bounds() const {
	if (stride <= 0) {
		return Shaded_rect{};
	}
	const int row0 = origin / stride;
	const int col0 = origin - row0 * stride;
	return Shaded_rect{-col0, -row0, width, height};
}

void Shaded_cpu::clear_rect(Shaded_scene_buffers& buf, const Shaded_rect& r) {
	fill_plane(buf, r, 0, shaded_face_none, shaded_id_none);
}

void Shaded_cpu::fill_plane(Shaded_scene_buffers& buf, const Shaded_rect& r, uint16_t height, uint8_t facing, uint16_t id) {
	for (int y = r.y; y < r.y + r.h; y++) {
		const Shaded_row row(buf, r.x, y);
		std::fill_n(row.height, r.w, height);
		std::memset(row.facing, facing, static_cast<size_t>(r.w));
		std::fill_n(row.ids, r.w, id);
	}
}

void Shaded_cpu::record_spans(
		Shaded_scene_buffers& buf, const Shaded_rect& clip, const Shaded_object_context& obj, const unsigned char* rle,
		int xoff, int yoff, int skip_from) {
	// Same format as Image_buffer8::paint_rle.
	const int    right  = clip.x + clip.w;
	const int    bottom = clip.y + clip.h;
	const uint8* in     = rle;
	int          scanlen;
	while ((scanlen = little_endian::Read2(in)) != 0) {
		const int encoded = scanlen & 1;
		scanlen           = scanlen >> 1;
		const int  scanx  = xoff + static_cast<sint16>(little_endian::Read2(in));
		const int  scany  = yoff + static_cast<sint16>(little_endian::Read2(in));
		const bool row_on = scany >= clip.y && scany < bottom;
		// Record count pixels from x; pix = their values (nullptr: all pixv).
		auto run = [&](int x, int count, const uint8* pix, int pixv) {
			if (!row_on) {
				return;
			}
			const int x0 = std::max(x, clip.x);
			const int x1 = std::min(x + count, right);
			if (x0 >= x1) {
				return;
			}
			const Shaded_row row(buf, x0, scany);
			const int        v = scany - obj.oy;
			for (int px = x0; px < x1; px++) {
				const int val = pix ? pix[px - x] : pixv;
				if (val >= skip_from && val <= 0xfe) {
					continue;    // Blended, so what is beneath shows.
				}
				const Shaded_box_hit hit    = Classify(obj, px - obj.ox, v);
				const int            height = obj.base_height + hit.dz_px * (shaded_height_units_per_lift / 4);
				const int            i      = px - x0;
				row.height[i]               = static_cast<uint16_t>(std::min(height, 0xffff));
				row.facing[i]               = hit.facing;
				row.ids[i]                  = obj.id;
			}
		};
		if (!encoded) {
			run(scanx, scanlen, in, 0);
			in += scanlen;
			continue;
		}
		for (int b = 0; b < scanlen;) {
			unsigned char bcnt   = Read1(in);
			const int     repeat = bcnt & 1;
			bcnt                 = bcnt >> 1;
			if (repeat) {
				run(scanx + b, bcnt, nullptr, Read1(in));
			} else {
				run(scanx + b, bcnt, in, 0);
				in += bcnt;
			}
			b += bcnt;
		}
	}
}

int Shaded_cpu::view_code(const Shaded_scene_buffers& buf, const Shaded_light_params& params, int index) {
	const uint8_t facing = buf.facing_map[index];
	switch (static_cast<Shaded_view>(params.view)) {
	case Shaded_view::height: {
		if ((facing & shaded_face_mask) == shaded_face_none) {
			return 0;
		}
		const float lifts = static_cast<float>(buf.height_map[index]) / params.height_units_per_lift;
		const int   code  = 1 + static_cast<int>(lifts / params.debug_height_range * (lut_size - 1));
		return std::min(code, lut_size - 1);
	}
	case Shaded_view::facing:
		return facing & (shaded_face_mask | shaded_face_clamped);
	case Shaded_view::object_id: {
		const unsigned id = buf.object_ids[index];
		if (id == shaded_id_none) {
			return 0;
		}
		return 1 + static_cast<int>((id * 2654435761u >> 16) % (lut_size - 1));
	}
	case Shaded_view::light:
		return 1 + buf.light_levels[index] * (lut_size - 2) / std::max(params.level_count - 1, 1);
	case Shaded_view::normal:
	case Shaded_view::unlit:
	case Shaded_view::count:
		break;
	}
	return 0;
}

void Shaded_cpu::build_bins(
		const Shaded_light_params& params, const std::vector<Shaded_light_record>& lights, std::vector<uint32_t>& offsets,
		std::vector<uint16_t>& list) {
	const int cells = std::max(params.bins_x, 0) * std::max(params.bins_y, 0);
	offsets.assign(static_cast<size_t>(cells) + 1, 0);
	list.clear();
	if (cells <= 0) {
		return;
	}
	// Bins each light covers: x0, y0, x1, y1 inclusive (x1 < x0: none).
	struct Span {
		int x0, y0, x1, y1;
	};

	std::vector<Span> spans(lights.size(), Span{0, 0, -1, -1});
	const auto        tiles = static_cast<float>(c_num_tiles);
	const auto        size  = static_cast<float>(params.bin_size);
	auto              wrap  = [tiles](float d) {
		d = std::fmod(d, tiles);
		if (d > tiles * 0.5f) {
			d -= tiles;
		} else if (d < -tiles * 0.5f) {
			d += tiles;
		}
		return d;
	};
	// Pixel range covered by world x (or y) in [lo, hi] tiles from the
	// scroll tile, at heights zlo..zhi lifts: px = 8 * w - scroll_px - 0.5 - 4 * z.
	auto cover = [&](float lo, float hi, int scroll_px, float zlo, float zhi, int count, int& c0, int& c1) {
		const float p0 = lo * params.pixels_per_tile - static_cast<float>(scroll_px) - 0.5f - zhi * params.pixels_per_lift;
		const float p1 = hi * params.pixels_per_tile - static_cast<float>(scroll_px) - 0.5f - zlo * params.pixels_per_lift;
		c0             = std::max(static_cast<int>(std::floor(p0 / size)), 0);
		c1             = std::min(static_cast<int>(std::floor(p1 / size)), count - 1);
	};
	for (size_t i = 0; i < lights.size(); i++) {
		const Shaded_light_record& l = lights[i];
		const float                r = l.radius * params.radius_scale;
		if (r <= 0.0f || (l.flags & shaded_light_off)) {
			continue;
		}
		// Heights the light can reach: within R in 3D, or the area band.
		float zlo, zhi;
		if (l.flags & shaded_light_area) {
			zlo = l.z - 1.0f;
			zhi = l.z + params.area_height;
		} else {
			zlo = l.z - r / params.tiles_per_lift;
			zhi = l.z + r / params.tiles_per_lift;
		}
		zlo               = std::max(zlo, 0.0f);
		zhi               = std::max(zhi, zlo);
		const float dx    = wrap(l.x - static_cast<float>(params.scroll_tx));
		const float dy    = wrap(l.y - static_cast<float>(params.scroll_ty));
		Span&       span  = spans[i];
		cover(dx - r, dx + r, params.scroll_px, zlo, zhi, params.bins_x, span.x0, span.x1);
		cover(dy - r, dy + r, params.scroll_py, zlo, zhi, params.bins_y, span.y0, span.y1);
		for (int y = span.y0; y <= span.y1; y++) {
			for (int x = span.x0; x <= span.x1; x++) {
				offsets[static_cast<size_t>(y * params.bins_x + x) + 1]++;
			}
		}
	}
	for (int c = 0; c < cells; c++) {
		offsets[static_cast<size_t>(c) + 1] += offsets[static_cast<size_t>(c)];
	}
	list.resize(offsets[static_cast<size_t>(cells)]);
	std::vector<uint32_t> fill(offsets.begin(), offsets.end() - 1);
	for (size_t i = 0; i < lights.size(); i++) {
		const Span& span = spans[i];
		for (int y = span.y0; y <= span.y1; y++) {
			for (int x = span.x0; x <= span.x1; x++) {
				list[fill[static_cast<size_t>(y * params.bins_x + x)]++] = static_cast<uint16_t>(i);
			}
		}
	}
}

void Shaded_cpu::light_rows(
		const Shaded_scene_buffers& buf, const Shaded_light_params& params, const Shaded_light_record* lights,
		const uint32_t* bin_offsets, const uint16_t* bin_lights, const Shaded_rect& r, int y0, int y1, uint8_t* levels) {
	if (params.bins_x <= 0 || params.bins_y <= 0) {
		return;
	}
	const int   top_level = std::max(params.level_count - 1, 1);
	const auto  tiles     = static_cast<float>(c_num_tiles);
	const float ambient   = params.ambient_level;
	const float inv_tile  = 1.0f / params.pixels_per_tile;
	const float inv_units = 1.0f / params.height_units_per_lift;
	const float lift_tile = params.pixels_per_lift * inv_tile;    // Tiles per lift of screen shift.
	const float inv_soft  = 1.0f / std::max(params.softness, 0.01f);
	const float facing    = params.facing_min;
	// Shortest way round the wrapped world.
	auto wrap = [tiles](float d) {
		if (d > tiles * 0.5f) {
			d -= tiles;
		} else if (d < -tiles * 0.5f) {
			d += tiles;
		}
		return d;
	};
	// Level for light l, dithered against world pixels so it does not crawl.
	auto quantise = [&](float l, int x, int y) {
		const float threshold = params.dither ? Shaded_bayer4(x + params.scroll_px, y + params.scroll_py) : 0.5f;
		const int   q         = static_cast<int>(std::floor(l * static_cast<float>(top_level) + threshold));
		return static_cast<uint8_t>(std::clamp(q, 0, top_level));
	};
	for (int y = y0; y < y1; y++) {
		const int   row    = buf.index(0, y);
		const int   bin_y  = std::clamp(y / params.bin_size, 0, params.bins_y - 1);
		const float base_x = static_cast<float>(params.scroll_tx) + static_cast<float>(r.x + params.scroll_px) * inv_tile + 0.5f * inv_tile;
		const float base_y = static_cast<float>(params.scroll_ty) + (static_cast<float>(y + params.scroll_py) + 0.5f) * inv_tile;
		for (int x = r.x; x < r.x + r.w; x++) {
			const int     idx  = row + x;
			const uint8_t face = buf.facing_map[idx] & shaded_face_mask;
			if (face == shaded_face_none) {
				levels[idx] = quantise(ambient, x, y);
				continue;
			}
			// World position as in Shaded_light::world_position, unwrapped.
			const float         wz   = static_cast<float>(buf.height_map[idx]) * inv_units;
			const float         px   = base_x + static_cast<float>(x - r.x) * inv_tile + wz * lift_tile;
			const float         py   = base_y + wz * lift_tile;
			const Shaded_normal n    = Shaded_face_normal(face);
			const int           id   = buf.object_ids[idx];
			const int           bin  = bin_y * params.bins_x + std::clamp(x / params.bin_size, 0, params.bins_x - 1);
			float               sum  = 0.0f;
			for (uint32_t k = bin_offsets[bin]; k < bin_offsets[bin + 1]; k++) {
				const Shaded_light_record& l = lights[bin_lights[k]];
				if (id != shaded_id_none && l.object_id == id) {
					sum += l.intensity;    // The light's own object.
					continue;
				}
				const float radius = l.radius * params.radius_scale;
				const float r2     = radius * radius;
				const float dx     = wrap(l.x - px);
				const float dy     = wrap(l.y - py);
				const float h2     = dx * dx + dy * dy;
				if (h2 >= r2) {
					continue;
				}
				if (l.flags & shaded_light_area) {
					if (wz >= l.z - 1.0f && wz <= l.z + params.area_height) {
						sum += l.intensity * Shaded_falloff(h2 / r2);
					}
					continue;
				}
				const float dz = (l.z - wz) * params.tiles_per_lift;
				const float d2 = h2 + dz * dz;
				if (d2 >= r2) {
					continue;
				}
				const float dot   = static_cast<float>(n.x) * dx + static_cast<float>(n.y) * dy + static_cast<float>(n.z) * dz;
				const float ndotl = d2 > 1e-6f ? dot / std::sqrt(d2) : 1.0f;
				if (ndotl <= 0.0f) {
					continue;
				}
				sum += l.intensity * Shaded_falloff(d2 / r2) * (facing + (1.0f - facing) * ndotl);
			}
			const float level = ambient + (1.0f - ambient) * (1.0f - std::exp(-sum * params.brightness * inv_soft));
			levels[idx]       = quantise(level, x, y);
		}
	}
}

void Shaded_cpu::resolve_view(
		const Shaded_scene_buffers& buf, const Shaded_light_params& params, const Shaded_rect& r, const uint8_t* lut,
		unsigned char* frame) {
	for (int y = r.y; y < r.y + r.h; y++) {
		const int idx = buf.index(r.x, y);
		for (int i = idx; i < idx + r.w; i++) {
			frame[i] = lut[view_code(buf, params, i)];
		}
	}
}

Shaded_volume Shaded_volume_for(const Game_object& obj) {
	// Hook for the rounded-volume stage: people, trees and other organic
	// shapes will return Shaded_volume::rounded.
	ignore_unused_variable_warning(obj);
	return Shaded_volume::box;
}

Shaded_light* Shaded_light::active = nullptr;

Shaded_light& Shaded_light::get() {
	static Shaded_light instance;
	return instance;
}

Shaded_light::Shaded_light() {
	read_config();
	bool enabled = lighting_on;
	if (const char* hour = std::getenv("EXULT_SHADED_HOUR"); hour && *hour) {
		pending_hour = std::clamp(std::atoi(hour), 0, 23);
	}
	if (const char* tele = std::getenv("EXULT_SHADED_TELEPORT"); tele && *tele) {
		pending_teleport = std::sscanf(tele, "%d,%d,%d", &teleport_to[0], &teleport_to[1], &teleport_to[2]) == 3;
		if (!pending_teleport) {
			std::cout << "SHADED: EXULT_SHADED_TELEPORT wants x,y,z, not " << tele << std::endl;
		}
	}
	if (const char* dir = std::getenv("EXULT_SHADED_DUMP"); dir && *dir) {
		dump_dir = dir;
		std::error_code ec;
		std::filesystem::create_directories(dump_dir, ec);
		if (ec) {
			std::cout << "SHADED: could not create dump folder " << dump_dir << ": " << ec.message() << std::endl;
			dump_dir.clear();
		} else {
			enabled     = true;
			lighting_on = true;
		}
	}
	const char* at = std::getenv("EXULT_SHADED_DUMP_AT");
	std::stringstream times(at && *at ? at : "6000");
	std::string       item;
	while (std::getline(times, item, ',')) {
		dump_at.push_back(std::atoi(item.c_str()));
	}
	std::sort(dump_at.begin(), dump_at.end());
	if (const char* view = std::getenv("EXULT_SHADED_VIEW")) {
		for (int v = 0; v < static_cast<int>(Shaded_view::count); v++) {
			if (std::string(view) == view_name(static_cast<Shaded_view>(v))) {
				params.view = v;
			}
		}
		if (std::string(view) == "lit") {    // The lit frame is the normal view.
			params.view = static_cast<int32_t>(Shaded_view::normal);
		}
	}
	// Light pass threads: config or EXULT_SHADED_THREADS, 0 = auto.
	if (const char* thr = std::getenv("EXULT_SHADED_THREADS"); thr && *thr) {
		params.threads = std::max(std::atoi(thr), 0);
	}
	if (params.threads <= 0) {
		params.threads = std::clamp(static_cast<int>(std::thread::hardware_concurrency()), 1, 8);
	}
	const char* tim = std::getenv("EXULT_SHADED_TIMING");
	timing          = tim && *tim && *tim != '0';
	record_on       = enabled;
	params.enabled  = enabled ? 1 : 0;
	if (record_on) {
		std::cout << "SHADED: scene record on" << (dump_dir.empty() ? "" : ", dumps in ") << dump_dir << std::endl;
	}
}

Shaded_light::~Shaded_light() {
	if (timing) {
		report_timing();
	}
}

Shaded_backend Shaded_light::backend() {
	switch (static_cast<Shaded_backend>(params.backend)) {
	case Shaded_backend::gpu:
		if (!warned_backend) {
			std::cout << "SHADED: GPU backend not built yet; using CPU" << std::endl;
			warned_backend = true;
		}
		params.backend = static_cast<int32_t>(Shaded_backend::cpu);
		break;
	case Shaded_backend::cpu:
		break;
	}
	return Shaded_backend::cpu;
}

const char* Shaded_light::view_name(Shaded_view view) {
	switch (view) {
	case Shaded_view::height:
		return "height";
	case Shaded_view::facing:
		return "facing";
	case Shaded_view::object_id:
		return "id";
	case Shaded_view::light:
		return "light";
	case Shaded_view::unlit:
		return "unlit";
	case Shaded_view::normal:
	case Shaded_view::count:
		break;
	}
	return "normal";
}

Shaded_view Shaded_light::cycle_view() {
	if (!record_on) {
		return Shaded_view::normal;
	}
	params.view = (params.view + 1) % static_cast<int>(Shaded_view::count);
	return static_cast<Shaded_view>(params.view);
}

/*
 *  Read config/video/shaded_light/...; percentages become fractions.
 */

void Shaded_light::read_config() {
	if (!config) {
		return;
	}
	const std::string base = "config/video/shaded_light/";
	bool              enabled;
	bool              old_enabled;    // Stage 1 form: config/video/shaded_light = yes.
	config->value(base + "enabled", enabled, false);
	config->value("config/video/shaded_light", old_enabled, false);
	lighting_on  = enabled || old_enabled;
	auto percent = [&](const char* key, int def) {
		int value;
		config->value(base + key, value, def);
		return static_cast<float>(std::max(value, 0)) / 100.0f;
	};
	params.brightness   = percent("brightness", 100);
	params.radius_scale = percent("radius", 100);
	params.ambient      = percent("ambient", 100);
	params.softness     = std::max(percent("softness", 100), 0.01f);
	params.facing_min   = std::min(percent("facing", 50), 1.0f);
	bool dither;
	config->value(base + "dither", dither, true);
	params.dither = dither ? 1 : 0;
	int threads;
	config->value(base + "threads", threads, 0);
	params.threads = std::max(threads, 0);    // 0 = auto, resolved later.
}

/*
 *  Headless debug hooks, once each.  The flags are cleared first, as both
 *  can paint again from inside.
 */

void Shaded_light::run_debug_hooks() {
	Game_window* gwin = Game_window::get_instance();
	if (pending_hour >= 0) {
		const int hour = pending_hour;
		pending_hour   = -1;
		Game_clock* clock = gwin->get_clock();
		clock->set_hour(hour);
		clock->set_minute(0);
		clock->set_ticks(0);
		clock->reset_palette();
		gwin->set_all_dirty();
		std::cout << "SHADED: hour set to " << hour << std::endl;
	}
	if (pending_teleport) {
		pending_teleport = false;
		gwin->teleport_party(Tile_coord(teleport_to[0], teleport_to[1], teleport_to[2]));
		std::cout << "SHADED: teleported to " << teleport_to[0] << "," << teleport_to[1] << "," << teleport_to[2] << std::endl;
	}
}

/*
 *  Read shaded_light.txt (bundled, then the patch folder's) when the game
 *  or mod has changed.  Without a file only the formula is used.
 */

void Shaded_light::load_table() {
	const Exult_Game  game = Game::get_game_type();
	const std::string key  = std::to_string(static_cast<int>(game)) + "/" + Game::get_gametitle() + "/" + Game::get_modtitle();
	if (table_loaded && key == table_game) {
		return;
	}
	table_loaded = true;
	table_game   = key;
	light_table.clear();
	int resource = -1;
	if (game == BLACK_GATE) {
		resource = EXULT_BG_FLX_SHADED_LIGHT_TXT;
	} else if (game == SERPENT_ISLE) {
		resource = EXULT_SI_FLX_SHADED_LIGHT_TXT;
	}
	std::array<std::string_view, 1> sections{"lights"};
	std::array readers = make_unique_array<Base_reader>(std::make_unique<Shaded_light_reader>(light_table));
	static_assert(sections.size() == readers.size());
	try {
		Read_text_data_file("shaded_light", readers, sections, false, game, resource);
	} catch (const file_open_exception& e) {
		std::cout << "SHADED: no light table (" << e.what() << "), using the default formula" << std::endl;
	}
	table_radius = 0;
	for (const auto& [id, entry] : light_table) {
		table_radius = std::max(table_radius, entry.radius);
	}
	std::cout << "SHADED: light table has " << light_table.size() << " rows" << std::endl;
}

/*
 *  The row for a light: exact (shape, frame & 31), then (shape, -1), else
 *  the formula from its light_data brightness.
 */

Shaded_light_entry Shaded_light::find_row(int shape, int frame, int brightness, Shaded_light_row& row) const {
	auto it = light_table.find(std::make_pair(shape, frame & 31));
	if (it != light_table.end()) {
		row = Shaded_light_row::exact;
		return it->second;
	}
	it = light_table.find(std::make_pair(shape, -1));
	if (it != light_table.end()) {
		row = Shaded_light_row::wildcard;
		return it->second;
	}
	row = Shaded_light_row::formula;
	return Shaded_default_light(brightness);
}

/*
 *  Add a light from source (the light object) placed on owner (the same
 *  object, or the party member carrying it).
 */

void Shaded_light::add_light(
		const Game_object& source, const Game_object& owner, const Shaded_light_entry& entry, Shaded_light_row row,
		bool carried) {
	if (lights.size() >= static_cast<size_t>(shaded_max_lights)) {
		if (!warned_lights) {
			warned_lights = true;
			std::cout << "SHADED: over " << shaded_max_lights << " lights in view, the rest are left out" << std::endl;
		}
		return;
	}
	Game_window* gwin = Game_window::get_instance();
	int          ox, oy;
	gwin->get_shape_location(&owner, ox, oy);
	const Tile_coord  t     = owner.get_tile();
	const Shape_info& info  = owner.get_info();
	const int         frame = owner.get_framenum();
	// (ox, oy) is the pixel up and left of the box's SE bottom corner, with
	// the smooth-scroll and barge offsets the painter used; undoing
	// Get_shape_location gives the corner exactly (tile tx + 1, ty + 1).
	const float lift_px  = static_cast<float>(t.tz) * params.pixels_per_lift;
	const float corner_x = static_cast<float>(gwin->get_scrolltx())
						   + (static_cast<float>(ox + gwin->get_scrolltx_lo() + 1) + lift_px) / params.pixels_per_tile;
	const float corner_y = static_cast<float>(gwin->get_scrollty())
						   + (static_cast<float>(oy + gwin->get_scrollty_lo() + 1) + lift_px) / params.pixels_per_tile;
	const auto  tiles    = static_cast<float>(c_num_tiles);
	const auto  ztiles   = static_cast<float>(info.get_3d_height());
	Shaded_light_record rec;
	rec.x = std::fmod(corner_x - static_cast<float>(info.get_3d_xtiles(frame)) * 0.5f + tiles, tiles);
	rec.y = std::fmod(corner_y - static_cast<float>(info.get_3d_ytiles(frame)) * 0.5f + tiles, tiles);
	if (carried) {    // About hand height.
		rec.z = static_cast<float>(t.tz) + 0.75f * ztiles;
	} else {
		rec.z = static_cast<float>(t.tz) + (entry.height >= 0 ? static_cast<float>(entry.height) : ztiles);
	}
	rec.radius    = static_cast<float>(entry.radius);
	rec.red       = static_cast<float>(entry.red) / 255.0f;
	rec.green     = static_cast<float>(entry.green) / 255.0f;
	rec.blue      = static_cast<float>(entry.blue) / 255.0f;
	rec.intensity = static_cast<float>(entry.brightness) / 100.0f;
	rec.flags     = (entry.flags & (shaded_light_area | shaded_light_off)) | (carried ? shaded_light_carried : 0);
	rec.object_id = shaded_id_none;    // Resolved once the paint has recorded it.
	rec.shape     = source.get_shapenum();
	rec.frame     = source.get_framenum();
	lights.push_back(rec);
	light_objects.push_back(&owner);
	light_sources.push_back(&source);
	light_rows.push_back(row);
}

/*
 *  The lights of this paint: lights in the chunks on screen and as far
 *  around as the largest radius reaches, then the party's carried lights.
 */

void Shaded_light::gather_lights() {
	Game_window* gwin = Game_window::get_instance();
	Game_map*    map  = gwin->get_map();
	lights.clear();
	light_objects.clear();
	light_sources.clear();
	light_rows.clear();
	const float max_radius = static_cast<float>(std::max(table_radius, Shaded_default_light(8).radius)) * params.radius_scale;
	const int   reach      = std::min(static_cast<int>(std::ceil(max_radius)), 32);
	const int   margin     = (reach + c_tiles_per_chunk - 1) / c_tiles_per_chunk;
	// Chunks as Game_render::paint_map walks them, widened by margin.
	auto range = [margin](int scroll, int pixels, int& start, int& count) {
		start          = (scroll - 1 + c_num_tiles) / c_tiles_per_chunk - c_num_chunks - margin;
		const int stop = 2 + (scroll + (pixels + c_tilesize - 2) / c_tilesize + c_tiles_per_chunk / 2) / c_tiles_per_chunk + margin;
		count          = std::min(stop - start, c_num_chunks);
	};
	int start_x, count_x, start_y, count_y;
	range(gwin->get_scrolltx(), gwin->get_width(), start_x, count_x);
	range(gwin->get_scrollty(), gwin->get_height(), start_y, count_y);
	auto wrap = [](int c) {
		return ((c % c_num_chunks) + c_num_chunks) % c_num_chunks;
	};
	const bool dungeon   = gwin->is_in_dungeon() != 0;
	const int  skip_lift = gwin->get_render_skip_lift();
	for (int j = 0; j < count_y; j++) {
		for (int i = 0; i < count_x; i++) {
			// Never creates chunks: one not yet read has no lights.
			const Map_chunk* chunk = map->get_chunk_unsafe(wrap(start_x + i), wrap(start_y + j));
			if (!chunk) {
				continue;
			}
			const auto& set = dungeon ? chunk->get_dungeon_lights() : chunk->get_non_dungeon_lights();
			for (const Game_object* obj : set) {
				const int frame = obj->get_framenum();
				const int b     = obj->get_info().get_object_light(frame);
				if (b <= 0 || obj->get_lift() >= skip_lift) {
					continue;
				}
				Shaded_light_row         row;
				const Shaded_light_entry entry = find_row(obj->get_shapenum(), frame, b, row);
				if (!(entry.flags & shaded_light_off)) {
					add_light(*obj, *obj, entry, row, false);
				}
			}
		}
	}
	// Carried lights: each readied light Actor::refigure_gear counts, at
	// the bearer; a formula light if none is found.
	static const Ready_type_Exult spots[]
			= {head, belt, lhand, lfinger, legs, feet, rfinger, rhand, torso, amulet, earrings, cloak, gloves};
	Actor*    party[9];
	const int count = gwin->get_party(party, 1);
	for (int p = 0; p < count; p++) {
		const Actor* actor = party[p];
		if (!actor || actor->get_light_source() <= 0 || actor->get_lift() >= skip_lift) {
			continue;
		}
		bool found = false;
		for (const auto spot : spots) {
			const Game_object* worn = actor->get_readied(spot);
			if (!worn) {
				continue;
			}
			const Shape_info& info  = worn->get_info();
			const int         rdy   = info.get_ready_type();
			const int         frame = worn->get_framenum();
			const int         b     = info.get_object_light(frame);
			if (!info.is_light_source() || b <= 0 || (spot == belt && (rdy == lhand || rdy == rhand || rdy == both_hands))) {
				continue;
			}
			found = true;
			Shaded_light_row         row;
			const Shaded_light_entry entry = find_row(worn->get_shapenum(), frame, b, row);
			if (!(entry.flags & shaded_light_off)) {
				add_light(*worn, *actor, entry, row, true);
			}
		}
		if (!found) {
			add_light(*actor, *actor, Shaded_default_light(actor->get_light_source()), Shaded_light_row::formula, true);
		}
	}
	// Hash of what the lighting depends on, positions to 1/16 tile.
	uint64_t hash = 14695981039346656037ull;
	for (const Shaded_light_record& rec : lights) {
		Hash_add(hash, static_cast<int32_t>(std::lround(rec.x * 16.0f)));
		Hash_add(hash, static_cast<int32_t>(std::lround(rec.y * 16.0f)));
		Hash_add(hash, static_cast<int32_t>(std::lround(rec.z * 16.0f)));
		Hash_add(hash, static_cast<int32_t>(std::lround(rec.radius * 16.0f)));
		Hash_add(hash, static_cast<int32_t>(std::lround(rec.intensity * 256.0f)));
		Hash_add(hash, rec.flags);
	}
	Hash_add(hash, static_cast<int32_t>(lights.size()));
	light_hash = hash;
}

bool Shaded_light::prepare_paint() {
	if (!record_on && !lighting_on) {
		return false;
	}
	run_debug_hooks();
	load_table();
	gather_lights();
	// Nothing is lit from the lights yet, so no paint needs widening.
	return false;
}

/*
 *  Match the buffers to the draw surface.  Returns false if the paint does
 *  not go into the draw surface (a redirected render target).
 */

bool Shaded_light::setup_buffers(Image_window8& win) {
	SDL_Surface*   surf = win.get_draw_surface();
	Image_buffer8* ib   = win.get_ib8();
	if (!surf || !surf->pixels || !ib || !ib->get_bits() || static_cast<int>(ib->get_line_width()) != surf->pitch) {
		return false;
	}
	const auto base   = reinterpret_cast<uintptr_t>(surf->pixels);
	const auto bits   = reinterpret_cast<uintptr_t>(ib->get_bits());
	const auto size   = static_cast<uintptr_t>(surf->pitch) * static_cast<uintptr_t>(surf->h);
	if (bits < base || bits - base >= size) {
		return false;
	}
	const int origin = static_cast<int>(bits - base);
	if (buffers.width != surf->w || buffers.height != surf->h || buffers.stride != surf->pitch || buffers.origin != origin) {
		const auto count = static_cast<size_t>(size);
		buffers.width    = surf->w;
		buffers.height   = surf->h;
		buffers.stride   = surf->pitch;
		buffers.origin   = origin;
		buffers.height_map.assign(count, 0);
		buffers.facing_map.assign(count, shaded_face_none);
		buffers.object_ids.assign(count, shaded_id_none);
		buffers.light_levels.assign(count, 0);
		buffers.objects.clear();
		object_index.clear();
		main_actor_id         = shaded_id_none;
		params.buffer_width   = buffers.width;
		params.buffer_height  = buffers.height;
		params.buffer_stride  = buffers.stride;
		params.buffer_origin  = buffers.origin;
	}
	return true;
}

void Shaded_light::begin_paint(Image_window8& win, bool full) {
	if (!record_on && !timing) {
		return;
	}
	paint_record_us = 0;
	// (Re)allocating the buffers is not part of the timed paint.
	const bool ready = record_on && setup_buffers(win);
	const Clock::time_point start = Clock::now();
	paint_start                   = start;
	if (!ready) {
		return;
	}
	Game_window*   gwin = Game_window::get_instance();
	Image_buffer8* ib   = win.get_ib8();
	int            cx, cy, cw, ch;
	ib->get_clip(cx, cy, cw, ch);
	clip = Intersect(Shaded_rect{cx, cy, cw, ch}, buffers.bounds());
	if (full) {
		buffers.objects.clear();
		object_index.clear();
		main_actor_id = shaded_id_none;
	} else if (gwin->get_scrolltx() != params.scroll_tx || gwin->get_scrollty() != params.scroll_ty) {
		// Scrolling copies the frame and paints only the new strip, so move
		// the record the same way.
		auto delta = [](int now, int before) {
			int d = (now - before) % c_num_tiles;
			if (d > c_num_tiles / 2) {
				d -= c_num_tiles;
			} else if (d < -c_num_tiles / 2) {
				d += c_num_tiles;
			}
			return d * c_tilesize;
		};
		shift(delta(gwin->get_scrolltx(), params.scroll_tx), delta(gwin->get_scrollty(), params.scroll_ty));
	}
	params.scroll_tx = gwin->get_scrolltx();
	params.scroll_ty = gwin->get_scrollty();
	params.scroll_px = gwin->get_scrolltx_lo();
	params.scroll_py = gwin->get_scrollty_lo();
	switch (backend()) {
	case Shaded_backend::gpu:    // backend() falls back to CPU.
	case Shaded_backend::cpu:
		Shaded_cpu::clear_rect(buffers, clip);
		break;
	}
	target = ib;
	active = this;
	if (timing) {
		paint_record_us += std::chrono::duration<double, std::micro>(Clock::now() - start).count();
	}
}

/*
 *  Move the record's game area by (-dx, -dy) pixels, as the frame was.
 */

void Shaded_light::shift(int dx, int dy) {
	const Shaded_rect area = buffers.bounds();
	if (std::abs(dx) >= area.w || std::abs(dy) >= area.h) {
		Shaded_cpu::clear_rect(buffers, area);
		return;
	}
	const int w     = area.w - std::abs(dx);
	const int src_x = area.x + std::max(dx, 0);
	const int dst_x = area.x + std::max(-dx, 0);
	auto      move_row = [&](int src_y, int dst_y) {
		const int from = buffers.index(src_x, src_y);
		const int to   = buffers.index(dst_x, dst_y);
		std::copy_n(buffers.height_map.begin() + from, w, buffers.height_map.begin() + to);
		std::copy_n(buffers.facing_map.begin() + from, w, buffers.facing_map.begin() + to);
		std::copy_n(buffers.object_ids.begin() + from, w, buffers.object_ids.begin() + to);
		std::copy_n(buffers.light_levels.begin() + from, w, buffers.light_levels.begin() + to);
	};
	const int rows = area.h - std::abs(dy);
	if (dy > 0 || (dy == 0 && dx > 0)) {    // Moving up or left: go forwards.
		for (int r = 0; r < rows; r++) {
			move_row(area.y + r + dy, area.y + r);
		}
	} else if (dy < 0) {
		for (int r = rows - 1; r >= 0; r--) {
			move_row(area.y + r, area.y + r - dy);
		}
	} else if (dx < 0) {    // Same row, moving right: copy_n can overlap.
		std::vector<uint16_t> h16(static_cast<size_t>(w));
		std::vector<uint8_t>  f8(static_cast<size_t>(w));
		std::vector<uint16_t> i16(static_cast<size_t>(w));
		std::vector<uint8_t>  l8(static_cast<size_t>(w));
		for (int r = 0; r < rows; r++) {
			const int from = buffers.index(src_x, area.y + r);
			const int to   = buffers.index(dst_x, area.y + r);
			std::copy_n(buffers.height_map.begin() + from, w, h16.begin());
			std::copy_n(buffers.facing_map.begin() + from, w, f8.begin());
			std::copy_n(buffers.object_ids.begin() + from, w, i16.begin());
			std::copy_n(buffers.light_levels.begin() + from, w, l8.begin());
			std::copy(h16.begin(), h16.end(), buffers.height_map.begin() + to);
			std::copy(f8.begin(), f8.end(), buffers.facing_map.begin() + to);
			std::copy(i16.begin(), i16.end(), buffers.object_ids.begin() + to);
			std::copy(l8.begin(), l8.end(), buffers.light_levels.begin() + to);
		}
	}
}

void Shaded_light::end_paint(Image_window8& win, bool full) {
	if (!record_on && !timing) {
		return;
	}
	const Clock::time_point stop = Clock::now();
	const bool              was_active = active == this;
	active                             = nullptr;
	target                             = nullptr;
	if (in_object) {
		end_object();
	}
	if (timing) {
		Image_buffer8* ib = win.get_ib8();
		int            cx, cy, cw, ch;
		ib->get_clip(cx, cy, cw, ch);
		const double us = std::chrono::duration<double, std::micro>(stop - paint_start).count();
		paints++;
		paint_pixels += static_cast<double>(cw) * ch;
		all_paint_us += us;
		all_record_us += paint_record_us;
		if (full) {
			game_width  = Game_window::get_instance()->get_width();
			game_height = Game_window::get_instance()->get_height();
			full_paints++;
			paint_map_us += us;
			record_us += paint_record_us;
		}
		// Every 5 s: Exult may quit without running static destructors.
		if (paints == 1) {
			last_report = stop;
		} else if (stop - last_report >= std::chrono::seconds(5)) {
			last_report = stop;
			report_timing();
		}
	}
	if (!was_active) {
		return;
	}
	params.object_count = static_cast<int32_t>(buffers.objects.size());
	finish_due          = true;
}

/*
 *  Light the rows of r into levels, in bands on the workers if threaded
 *  and r is big enough.  Returns the number of bands.
 */

int Shaded_light::light_pass(const Shaded_rect& r, uint8_t* levels, bool threaded) {
	if (r.w <= 0 || r.h <= 0) {
		return 0;
	}
	auto band = [&](int b, int bands) {
		const int y0 = r.y + r.h * b / bands;
		const int y1 = r.y + r.h * (b + 1) / bands;
		Shaded_cpu::light_rows(
				buffers, params, lights.data(), buffers.bin_offsets.data(), buffers.bin_lights.data(), r, y0, y1, levels);
	};
	const int bands = threaded && params.threads > 1 && r.w * r.h >= 4096 ? std::min(params.threads, r.h) : 1;
	if (bands <= 1) {
		band(0, 1);
		return 1;
	}
	if (!workers || workers->size() != params.threads) {
		workers = std::make_unique<Shaded_workers>(params.threads);
	}
	workers->run(bands, [&](int b) {
		band(b, bands);
	});
	return bands;
}

void Shaded_light::finish_paint(Image_window8& win, bool full) {
	if (!finish_due) {
		return;
	}
	finish_due                    = false;
	const Clock::time_point start = Clock::now();
	// The lights' own objects, as recorded in this frame (0 = not painted).
	for (size_t i = 0; i < lights.size(); i++) {
		const auto it         = object_index.find(light_objects[i]);
		const int  id         = it != object_index.end() ? it->second : shaded_id_none;
		lights[i].object_id   = id == shaded_id_overflow ? shaded_id_none : id;
	}
	Game_window* gwin  = Game_window::get_instance();
	params.light_count = static_cast<int32_t>(lights.size());
	params.bins_x      = (gwin->get_width() + params.bin_size - 1) / params.bin_size;
	params.bins_y      = (gwin->get_height() + params.bin_size - 1) / params.bin_size;
	int bands          = 0;
	switch (backend()) {
	case Shaded_backend::gpu:    // backend() falls back to CPU.
	case Shaded_backend::cpu:
		Shaded_cpu::build_bins(params, lights, buffers.bin_offsets, buffers.bin_lights);
		bands = light_pass(clip, buffers.light_levels.data(), true);
		break;
	}
	const Clock::time_point stop = Clock::now();
	if (timing) {
		const double us = std::chrono::duration<double, std::micro>(stop - start).count();
		light_passes++;
		all_light_us += us;
		if (full) {
			full_light_passes++;
			full_light_us += us;
		}
	}
	// Due a dump?  Wanted on a full paint; full paints can be rare (the
	// game repaints only dirty areas), so after a second's wait a partial
	// paint does: the record outside its clip is still that of the frame.
	bool due = false;
	if (!dump_dir.empty()) {
		if (!clock_started && full) {
			clock_started = true;
			first_paint   = stop;
		}
		if (clock_started && dump_next < dump_at.size()) {
			const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - first_paint).count();
			due           = ms >= dump_at[dump_next] && (full || ms >= dump_at[dump_next] + 1000);
		}
	}
	if (due) {
		// The threaded pass must give what one thread does.
		if (bands > 1) {
			std::vector<uint8_t> single(buffers.light_levels);
			light_pass(clip, single.data(), false);
			bool same = true;
			for (int y = clip.y; y < clip.y + clip.h && same; y++) {
				const int idx = buffers.index(clip.x, y);
				same          = std::equal(
						 single.begin() + idx, single.begin() + idx + clip.w, buffers.light_levels.begin() + idx);
			}
			thread_check = std::string("check threads identical ") + (same ? "yes" : "no") + " ("
						   + std::to_string(params.threads) + " threads, " + std::to_string(bands) + " bands)";
		} else {
			thread_check = "check threads not compared (" + std::to_string(params.threads) + " threads, "
						   + std::to_string(bands) + " band)";
		}
		dump(win, static_cast<int>(dump_next), false, full);
		thread_check.clear();
	}
	const auto view = static_cast<Shaded_view>(params.view);
	if (view != Shaded_view::normal && view != Shaded_view::unlit) {
		uint8_t lut[lut_size];
		build_lut(lut);
		auto* frame = static_cast<unsigned char*>(win.get_draw_surface()->pixels);
		switch (backend()) {
		case Shaded_backend::gpu:    // backend() falls back to CPU.
		case Shaded_backend::cpu:
			Shaded_cpu::resolve_view(buffers, params, clip, lut, frame);
			break;
		}
		if (due) {
			dump(win, static_cast<int>(dump_next), true, full);
		}
	}
	if (due) {
		dump_next++;
		if (timing) {
			report_timing();
		}
	}
}

void Shaded_light::begin_object(const Game_object& obj, int ox, int oy) {
	if (active != this) {
		return;
	}
	const Clock::time_point start = timing ? Clock::now() : Clock::time_point();
	const Shape_info&       info  = obj.get_info();
	const int               frame = obj.get_framenum();
	const Tile_coord        t     = obj.get_tile();
	const Shaded_object_record rec{
			t.tx, t.ty, t.tz, info.get_3d_xtiles(frame), info.get_3d_ytiles(frame), info.get_3d_height(), obj.get_shapenum(), frame};
	auto [it, inserted] = object_index.try_emplace(&obj, shaded_id_overflow);
	if (inserted && buffers.objects.size() < shaded_id_overflow - 1) {
		buffers.objects.push_back(rec);
		it->second = static_cast<uint16_t>(buffers.objects.size());
	} else if (!inserted && it->second != shaded_id_overflow) {
		buffers.objects[it->second - 1] = rec;    // Moved or changed frame.
	}
	object.ox          = ox;
	object.oy          = oy;
	object.xpx         = rec.xtiles * c_tilesize;
	object.ypx         = rec.ytiles * c_tilesize;
	object.hpx         = rec.ztiles * (c_tilesize / 2);
	object.base_height = rec.tz * shaded_height_units_per_lift;
	object.id          = it->second;
	object.volume      = Shaded_volume_for(obj);
	if (&obj == Game_window::get_instance()->get_main_actor()) {
		main_actor_id = object.id;
	}
	in_object = true;
	Shape_manager::get_instance()->set_paint_listener(this);
	if (timing) {
		paint_record_us += std::chrono::duration<double, std::micro>(Clock::now() - start).count();
	}
}

void Shaded_light::end_object() {
	Shape_manager::get_instance()->set_paint_listener(nullptr);
	in_object = false;
}

void Shaded_light::shape_painted(int xoff, int yoff, Shape_frame& shape, bool translucent, int xfcnt) {
	// Only shapes painted for the current world object, into the frame.
	if (!in_object || active != this || Game_window::get_instance()->get_win()->get_ib8() != target) {
		return;
	}
	const Clock::time_point start = timing ? Clock::now() : Clock::time_point();
	switch (backend()) {
	case Shaded_backend::gpu:    // backend() falls back to CPU.
	case Shaded_backend::cpu:
		if (shape.is_rle()) {
			Shaded_cpu::record_spans(buffers, clip, object, shape.get_data(), xoff, yoff, translucent ? 0xff - xfcnt : 0x100);
		} else {    // An 8x8 flat tile, drawn up and left of its origin.
			const Shaded_rect r = Intersect(Shaded_rect{xoff - c_tilesize, yoff - c_tilesize, c_tilesize, c_tilesize}, clip);
			Shaded_cpu::fill_plane(
					buffers, r, static_cast<uint16_t>(object.base_height), shaded_face_ground, object.id);
		}
		break;
	}
	if (timing) {
		paint_record_us += std::chrono::duration<double, std::micro>(Clock::now() - start).count();
	}
}

void Shaded_light::record_plane(int x, int y, int w, int h, int lift, uint8_t face) {
	if (active != this) {
		return;
	}
	const Clock::time_point start = timing ? Clock::now() : Clock::time_point();
	const Shaded_rect       r     = Intersect(Shaded_rect{x, y, w, h}, clip);
	switch (backend()) {
	case Shaded_backend::gpu:    // backend() falls back to CPU.
	case Shaded_backend::cpu:
		Shaded_cpu::fill_plane(
				buffers, r, static_cast<uint16_t>(std::min(lift * shaded_height_units_per_lift, 0xffff)), face, shaded_id_none);
		break;
	}
	if (timing) {
		paint_record_us += std::chrono::duration<double, std::micro>(Clock::now() - start).count();
	}
}

void Shaded_light::world_position(int x, int y, float& wx, float& wy, float& wz) const {
	const int idx = buffers.index(x, y);
	wz            = static_cast<float>(buffers.height_map[idx]) / params.height_units_per_lift;
	const float lift_px = wz * params.pixels_per_lift;
	wx = static_cast<float>(params.scroll_tx)
		 + (static_cast<float>(x + params.scroll_px) + 0.5f + lift_px) / params.pixels_per_tile;
	wy = static_cast<float>(params.scroll_ty)
		 + (static_cast<float>(y + params.scroll_py) + 0.5f + lift_px) / params.pixels_per_tile;
	const auto tiles = static_cast<float>(c_num_tiles);
	wx               = std::fmod(wx + tiles, tiles);
	wy               = std::fmod(wy + tiles, tiles);
}

/*
 *  Palette index for each false colour code of the current view.
 */

void Shaded_light::build_lut(uint8_t* lut) const {
	const Palette* pal   = Game_window::get_instance()->get_pal();
	const int      scale = pal->get_max_val();
	for (int code = 0; code < lut_size; code++) {
		Rgb col{0, 0, 0};
		switch (static_cast<Shaded_view>(params.view)) {
		case Shaded_view::height:
			if (code > 0) {
				col = Height_colour(static_cast<float>(code - 1) / (lut_size - 2));
			}
			break;
		case Shaded_view::facing:
			col = Facing_colour(static_cast<uint8_t>(code));
			break;
		case Shaded_view::object_id:
			if (code > 0) {
				col = Id_colour(static_cast<unsigned>(code) * 7919u);
			}
			break;
		case Shaded_view::light:
			if (code > 0) {
				const int grey = (code - 1) * 255 / (lut_size - 2);
				col            = Rgb{grey, grey, grey};
			}
			break;
		case Shaded_view::normal:
		case Shaded_view::unlit:
		case Shaded_view::count:
			break;
		}
		const int index = pal->find_color(col.r * scale / 255, col.g * scale / 255, col.b * scale / 255);
		lut[code]       = static_cast<uint8_t>(std::max(index, 0));
	}
}

/*
 *  Write the record (or, with view, the resolved debug frame) as images.
 */

void Shaded_light::dump(Image_window8& win, int number, bool view, bool full) {
	Game_window*  gwin   = Game_window::get_instance();
	const int     gw     = gwin->get_width();
	const int     gh     = gwin->get_height();
	const auto*   frame  = static_cast<const unsigned char*>(win.get_draw_surface()->pixels);
	const Palette* pal   = gwin->get_pal();
	const int     maxval = std::max(pal->get_max_val(), 1);
	const std::string prefix = dump_dir + "/shaded-" + std::to_string(number) + "-";
	auto pixel_rgb           = [](unsigned char* out, const Rgb& c) {
		out[0] = static_cast<unsigned char>(c.r);
		out[1] = static_cast<unsigned char>(c.g);
		out[2] = static_cast<unsigned char>(c.b);
	};
	auto image = [&](const char* name, auto&& colour) {
		std::vector<unsigned char> rgb(static_cast<size_t>(gw) * gh * 3);
		for (int y = 0; y < gh; y++) {
			for (int x = 0; x < gw; x++) {
				pixel_rgb(&rgb[(static_cast<size_t>(y) * gw + x) * 3], colour(buffers.index(x, y)));
			}
		}
#ifdef HAVE_PNG_H
		if (!Write_png(prefix + name + ".png", gw, gh, false, rgb)) {
			std::cout << "SHADED: could not write " << prefix << name << ".png" << std::endl;
		}
#else
		ignore_unused_variable_warning(name);
#endif
	};
	auto frame_colour = [&](int idx) {
		const int pix = frame[idx];
		return Rgb{pal->get_red(pix) * 255 / maxval, pal->get_green(pix) * 255 / maxval, pal->get_blue(pix) * 255 / maxval};
	};
#ifndef HAVE_PNG_H
	std::cout << "SHADED: PNG support not built; dump skipped" << std::endl;
	ignore_unused_variable_warning(image, frame_colour, view, full);
	return;
#else
	if (view) {
		image("view", frame_colour);
		std::cout << "SHADED: dump " << number << " view " << view_name(static_cast<Shaded_view>(params.view)) << " written to "
				  << prefix << "view.png" << std::endl;
		return;
	}
	image("normal", frame_colour);
	image("height", [&](int idx) {
		if ((buffers.facing_map[idx] & shaded_face_mask) == shaded_face_none) {
			return Rgb{0, 0, 0};
		}
		return Height_colour(
				static_cast<float>(buffers.height_map[idx]) / params.height_units_per_lift / params.debug_height_range);
	});
	image("facing", [&](int idx) {
		return Facing_colour(buffers.facing_map[idx]);
	});
	image("id", [&](int idx) {
		return Id_colour(buffers.object_ids[idx]);
	});
	const int top_level = std::max(params.level_count - 1, 1);
	image("light", [&](int idx) {
		const int grey = buffers.light_levels[idx] * 255 / top_level;
		return Rgb{grey, grey, grey};
	});
	std::vector<unsigned char> grey(static_cast<size_t>(gw) * gh * 2);
	for (int y = 0; y < gh; y++) {
		for (int x = 0; x < gw; x++) {
			const uint16_t h   = buffers.height_map[buffers.index(x, y)];
			const size_t   out = (static_cast<size_t>(y) * gw + x) * 2;
			grey[out]          = static_cast<unsigned char>(h >> 8);
			grey[out + 1]      = static_cast<unsigned char>(h & 0xff);
		}
	}
	if (!Write_png(prefix + "height16.png", gw, gh, true, grey)) {
		std::cout << "SHADED: could not write " << prefix << "height16.png" << std::endl;
	}
	// Facts for checking the geometry: counts per face, and the avatar's
	// pixels rebuilt into world positions, which should lie in its box.
	long  faces[8] = {};
	long  avatar_px = 0, avatar_clamped = 0;
	long  box_px = 0, box_outside = 0;    // Unclamped object pixels.
	long  clamped_px = 0;
	float lo[3] = {1e9f, 1e9f, 1e9f}, hi[3] = {-1e9f, -1e9f, -1e9f};
	for (int y = 0; y < gh; y++) {
		for (int x = 0; x < gw; x++) {
			const int idx = buffers.index(x, y);
			faces[buffers.facing_map[idx] & shaded_face_mask]++;
			clamped_px += (buffers.facing_map[idx] & shaded_face_clamped) ? 1 : 0;
			const uint16_t id = buffers.object_ids[idx];
			if (id != shaded_id_none && id != shaded_id_overflow && !(buffers.facing_map[idx] & shaded_face_clamped)) {
				// Rebuilt position must lie in the object's box (within
				// half a pixel; world x/y wrap is ignored here).
				const Shaded_object_record& o = buffers.objects[id - 1];
				float                       w[3];
				world_position(x, y, w[0], w[1], w[2]);
				constexpr float e = 0.07f;
				box_px++;
				if (w[0] < static_cast<float>(o.tx + 1 - o.xtiles) - e || w[0] > static_cast<float>(o.tx + 1) + e
					|| w[1] < static_cast<float>(o.ty + 1 - o.ytiles) - e || w[1] > static_cast<float>(o.ty + 1) + e
					|| w[2] < static_cast<float>(o.tz) - e || w[2] > static_cast<float>(o.tz + o.ztiles) + e) {
					box_outside++;
				}
			}
			if (main_actor_id == shaded_id_none || buffers.object_ids[idx] != main_actor_id) {
				continue;
			}
			avatar_px++;
			if (buffers.facing_map[idx] & shaded_face_clamped) {
				avatar_clamped++;
				continue;
			}
			float w[3];
			world_position(x, y, w[0], w[1], w[2]);
			for (int i = 0; i < 3; i++) {
				lo[i] = std::min(lo[i], w[i]);
				hi[i] = std::max(hi[i], w[i]);
			}
		}
	}
	std::ofstream info(prefix + "info.txt");
	info << "game " << gw << "x" << gh << "\n"
		 << "paint " << (full ? "full" : "partial (frame outside its clip may hold earlier sprites or text)") << "\n"
		 << "buffer " << buffers.width << "x" << buffers.height << " stride " << buffers.stride << " origin " << buffers.origin
		 << "\n"
		 << "scroll tile " << params.scroll_tx << "," << params.scroll_ty << " px " << params.scroll_px << ","
		 << params.scroll_py << "\n"
		 << "objects " << buffers.objects.size() << "\n"
		 << "pixels none " << faces[shaded_face_none] << " ground " << faces[shaded_face_ground] << " top "
		 << faces[shaded_face_top] << " south " << faces[shaded_face_south] << " east " << faces[shaded_face_east] << "\n"
		 << "object pixels (unclamped) " << box_px << " rebuilt outside their box " << box_outside << ", clamped "
		 << clamped_px << "\n";
	if (main_actor_id != shaded_id_none && main_actor_id != shaded_id_overflow) {
		const Shaded_object_record& a = buffers.objects[main_actor_id - 1];
		// The box spans tiles (tx + 1 - xtiles .. tx + 1) and lifts tz .. tz + ztiles.
		const float box_lo[3] = {static_cast<float>(a.tx + 1 - a.xtiles), static_cast<float>(a.ty + 1 - a.ytiles),
								 static_cast<float>(a.tz)};
		const float box_hi[3] = {static_cast<float>(a.tx + 1), static_cast<float>(a.ty + 1), static_cast<float>(a.tz + a.ztiles)};
		bool        inside    = avatar_px > avatar_clamped;
		for (int i = 0; i < 3; i++) {
			// Pixel centres can sit half a pixel (1/16 tile) past the box.
			inside = inside && lo[i] >= box_lo[i] - 0.07f && hi[i] <= box_hi[i] + 0.07f;
		}
		info << "avatar id " << main_actor_id << " shape " << a.shape << " frame " << a.frame << " tile " << a.tx << "," << a.ty
			 << "," << a.tz << " box " << a.xtiles << "x" << a.ytiles << "x" << a.ztiles << "\n"
			 << "avatar pixels " << avatar_px << " clamped " << avatar_clamped << "\n"
			 << "avatar world x " << lo[0] << ".." << hi[0] << " y " << lo[1] << ".." << hi[1] << " z " << lo[2] << ".." << hi[2]
			 << "\n"
			 << "avatar box x " << box_lo[0] << ".." << box_hi[0] << " y " << box_lo[1] << ".." << box_hi[1] << " z "
			 << box_lo[2] << ".." << box_hi[2] << "\n"
			 << "avatar inside box " << (inside ? "yes" : "no") << "\n";
	} else {
		info << "avatar not recorded\n";
	}
	// The light list, with the pixels its lit object painted in this frame
	// (0 = not painted: hidden, or off screen).
	std::vector<long> id_pixels(buffers.objects.size() + 1, 0);
	for (int y = 0; y < gh; y++) {
		for (int x = 0; x < gw; x++) {
			const uint16_t id = buffers.object_ids[buffers.index(x, y)];
			if (id != shaded_id_overflow && id < id_pixels.size()) {
				id_pixels[id]++;
			}
		}
	}
	// Light levels over the game area, and the bins.
	long level_counts[shaded_light_levels] = {};
	for (int y = 0; y < gh; y++) {
		for (int x = 0; x < gw; x++) {
			level_counts[std::min<int>(buffers.light_levels[buffers.index(x, y)], shaded_light_levels - 1)]++;
		}
	}
	info << "light levels " << params.level_count << " ambient " << params.ambient_level << " dither " << params.dither
		 << " softness " << params.softness << " facing " << params.facing_min << " brightness " << params.brightness
		 << " radius " << params.radius_scale << "\n"
		 << "light level pixels";
	for (const long n : level_counts) {
		info << " " << n;
	}
	info << "\n"
		 << "bins " << params.bins_x << "x" << params.bins_y << " of " << params.bin_size << " px, entries "
		 << buffers.bin_lights.size() << "\n";
	if (!thread_check.empty()) {
		info << thread_check << "\n";
	}
	info << "lights " << lights.size() << " hash " << std::hex << light_hash << std::dec << "\n";
	for (size_t i = 0; i < lights.size(); i++) {
		const Shaded_light_record& rec = lights[i];
		const Tile_coord           t   = light_objects[i]->get_tile();
		const auto                 it  = object_index.find(light_objects[i]);
		const unsigned             id  = it != object_index.end() ? it->second : shaded_id_none;
		const long                 px  = id != shaded_id_none && id < id_pixels.size() ? id_pixels[id] : 0;
		info << "light " << i << " shape " << rec.shape << " frame " << rec.frame << " tile " << t.tx << "," << t.ty << ","
			 << t.tz << " world " << rec.x << "," << rec.y << "," << rec.z << " radius " << rec.radius << " intensity "
			 << rec.intensity << " flags " << rec.flags << " source "
			 << ((rec.flags & shaded_light_carried) ? "carried " : "") << Row_name(light_rows[i]) << " id " << id
			 << " pixels " << px << "\n";
	}
	std::cout << "SHADED: dump " << number << " written to " << prefix << "*.png (" << buffers.objects.size() << " objects)"
			  << std::endl;
#endif
}

void Shaded_light::report_timing() const {
	const double n  = full_paints > 0 ? static_cast<double>(full_paints) : 1.0;
	const double np = paints > 0 ? static_cast<double>(paints) : 1.0;
	std::cout << "SHADED-TIMING: record=" << (record_on ? "on" : "off") << " game=" << game_width << "x" << game_height
			  << " full_paints=" << full_paints << " paint_map_avg_us=" << paint_map_us / n << " record_avg_us=" << record_us / n
			  << " all_paints=" << paints << " avg_pixels=" << paint_pixels / np << " all_paint_map_avg_us=" << all_paint_us / np
			  << " all_record_avg_us=" << all_record_us / np
			  << " light_avg_us=" << full_light_us / static_cast<double>(std::max(full_light_passes, 1LL))
			  << " all_light_avg_us=" << all_light_us / static_cast<double>(std::max(light_passes, 1LL))
			  << " light_passes=" << light_passes << " threads=" << params.threads << " lights=" << params.light_count
			  << std::endl;
}
