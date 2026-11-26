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
#include "exult.h"
#include "gamewin.h"
#include "ibuf8.h"
#include "ignore_unused_variable_warning.h"
#include "iwin8.h"
#include "palette.h"
#include "shapeinf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

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
	case Shaded_view::normal:
	case Shaded_view::count:
		break;
	}
	return 0;
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
	bool enabled = false;
	if (config) {
		config->value("config/video/shaded_light", enabled, false);
	}
	if (const char* dir = std::getenv("EXULT_SHADED_DUMP"); dir && *dir) {
		dump_dir = dir;
		std::error_code ec;
		std::filesystem::create_directories(dump_dir, ec);
		if (ec) {
			std::cout << "SHADED: could not create dump folder " << dump_dir << ": " << ec.message() << std::endl;
			dump_dir.clear();
		} else {
			enabled = true;
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
		for (int r = 0; r < rows; r++) {
			const int from = buffers.index(src_x, area.y + r);
			const int to   = buffers.index(dst_x, area.y + r);
			std::copy_n(buffers.height_map.begin() + from, w, h16.begin());
			std::copy_n(buffers.facing_map.begin() + from, w, f8.begin());
			std::copy_n(buffers.object_ids.begin() + from, w, i16.begin());
			std::copy(h16.begin(), h16.end(), buffers.height_map.begin() + to);
			std::copy(f8.begin(), f8.end(), buffers.facing_map.begin() + to);
			std::copy(i16.begin(), i16.end(), buffers.object_ids.begin() + to);
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
		dump(win, static_cast<int>(dump_next), false, full);
	}
	if (static_cast<Shaded_view>(params.view) != Shaded_view::normal) {
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
		case Shaded_view::normal:
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
			  << " all_record_avg_us=" << all_record_us / np << std::endl;
}
