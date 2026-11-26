/*
 *  shadedweather.cc - Weather in the "shaded light" renderer.
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

#include "shadedweather.h"

#include "endianio.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

namespace {
	// Visit the pixels of an RLE frame painted at its origin, as
	// Image_buffer8::paint_rle reads it: visit(x, y) per pixel, x and y
	// relative to the origin.
	template <typename Visit>
	void For_rle_pixels(const unsigned char* rle, Visit&& visit) {
		const uint8* in = rle;
		int          scanlen;
		while ((scanlen = little_endian::Read2(in)) != 0) {
			const int encoded = scanlen & 1;
			scanlen           = scanlen >> 1;
			const int scanx   = static_cast<sint16>(little_endian::Read2(in));
			const int scany   = static_cast<sint16>(little_endian::Read2(in));
			if (!encoded) {
				for (int i = 0; i < scanlen; i++) {
					visit(scanx + i, scany);
				}
				in += scanlen;
				continue;
			}
			for (int b = 0; b < scanlen;) {
				unsigned char bcnt   = Read1(in);
				const int     repeat = bcnt & 1;
				bcnt                 = bcnt >> 1;
				for (int i = 0; i < bcnt; i++) {
					visit(scanx + b + i, scany);
				}
				in += repeat ? 1 : bcnt;
				b += bcnt;
			}
		}
	}

	// One box blur pass of radius r over a w x h byte image, along x
	// (step 1) or y (step w), rounding to nearest.
	void Box_blur(std::vector<uint8_t>& img, int w, int h, int r, bool along_x) {
		const int            n     = along_x ? w : h;
		const int            lines = along_x ? h : w;
		const int            step  = along_x ? 1 : w;
		const int            width = 2 * r + 1;
		std::vector<uint8_t> line(static_cast<size_t>(n));
		for (int l = 0; l < lines; l++) {
			const size_t base = along_x ? static_cast<size_t>(l) * static_cast<size_t>(w) : static_cast<size_t>(l);
			for (int i = 0; i < n; i++) {
				line[static_cast<size_t>(i)] = img[base + static_cast<size_t>(i) * static_cast<size_t>(step)];
			}
			int sum = 0;
			for (int i = -r; i <= r; i++) {
				sum += i >= 0 && i < n ? line[static_cast<size_t>(i)] : 0;
			}
			for (int i = 0; i < n; i++) {
				img[base + static_cast<size_t>(i) * static_cast<size_t>(step)] = static_cast<uint8_t>((sum + width / 2) / width);
				const int out = i - r;
				const int in  = i + r + 1;
				sum -= out >= 0 ? line[static_cast<size_t>(out)] : 0;
				sum += in < n ? line[static_cast<size_t>(in)] : 0;
			}
		}
	}

	std::string Fmt(double v, int prec = 3) {
		std::ostringstream out;
		out << std::fixed << std::setprecision(prec) << v;
		return out.str();
	}
}    // namespace

/*
 *  The sky map: chamfer distances (1, sqrt 2) from the open columns, in
 *  two forward and backward raster sweeps through columns clear at lift 1.
 */

void Shaded_cpu::build_sky_open(const Shaded_grid_volume& vol, float fade_tiles, std::vector<uint8_t>& open) {
	const int w = vol.size_x;
	const int h = vol.size_y;
	open.assign(static_cast<size_t>(std::max(w, 0)) * static_cast<size_t>(std::max(h, 0)), 0);
	if (w <= 0 || h <= 0 || vol.sky.size() < open.size()) {
		return;
	}
	constexpr float      unreached  = std::numeric_limits<float>::max();
	constexpr float      diag = 1.41421356f;
	const int            lift = std::min(1, vol.size_z - 1);
	std::vector<float>   dist(open.size(), unreached);
	std::vector<uint8_t> pass(open.size(), 0);
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			const size_t i = static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x);
			dist[i]        = vol.sky[i] == 0 ? 0.0f : unreached;
			pass[i]        = vol.opacity(x, y, lift) < shaded_grid_sky_threshold ? 1 : 0;
		}
	}
	auto at = [&](int x, int y) {
		return x < 0 || y < 0 || x >= w || y >= h ? unreached : dist[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)];
	};
	auto relax = [&](int x, int y, int sx, int sy) {
		const size_t i = static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x);
		if (dist[i] == 0.0f || !pass[i]) {
			return;
		}
		dist[i] = std::min({dist[i], at(x - sx, y) + 1.0f, at(x, y - sy) + 1.0f, at(x - sx, y - sy) + diag,
							at(x + sx, y - sy) + diag});
	};
	// Twice: a path round a corner may need a second pair of sweeps.
	for (int round = 0; round < 2; round++) {
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				relax(x, y, 1, 1);
			}
		}
		for (int y = h - 1; y >= 0; y--) {
			for (int x = w - 1; x >= 0; x--) {
				relax(x, y, -1, -1);
			}
		}
	}
	for (size_t i = 0; i < open.size(); i++) {
		if (dist[i] == 0.0f) {
			open[i] = 255;
		} else if (fade_tiles > 0.0f && dist[i] < fade_tiles) {
			open[i] = static_cast<uint8_t>(std::clamp(std::lround(255.0f * (1.0f - dist[i] / fade_tiles)), 0L, 255L));
		}
	}
}

/*
 *  A cloud frame's soft mask.  The padding is twice the blur radius, the
 *  reach of two box passes, so the soft edge is never cut off.
 */

void Shaded_cpu::build_cloud_mask(
		const unsigned char* rle, int xleft, int yabove, int w, int h, int blur, std::vector<uint8_t>& data,
		Shaded_cloud_mask& mask) {
	blur          = std::max(blur, 0);
	const int pad = 2 * blur;
	mask.x0       = -xleft - pad;
	mask.y0       = -yabove - pad;
	mask.w        = std::max(w, 0) + 2 * pad;
	mask.h        = std::max(h, 0) + 2 * pad;
	mask.offset   = static_cast<uint32_t>(data.size());
	std::vector<uint8_t> img(static_cast<size_t>(mask.w) * static_cast<size_t>(mask.h), 0);
	For_rle_pixels(rle, [&](int x, int y) {
		const int mx = x - mask.x0;
		const int my = y - mask.y0;
		if (mx >= 0 && my >= 0 && mx < mask.w && my < mask.h) {
			img[static_cast<size_t>(my) * static_cast<size_t>(mask.w) + static_cast<size_t>(mx)] = 255;
		}
	});
	if (blur > 0) {
		for (int pass = 0; pass < 2; pass++) {
			Box_blur(img, mask.w, mask.h, blur, true);
			Box_blur(img, mask.w, mask.h, blur, false);
		}
	}
	data.insert(data.end(), img.begin(), img.end());
}

/*
 *  Synthetic tests: a roofed room with a doorway, a closed room, and a
 *  hand-made square cloud frame.
 */

int Shaded_weather_selftest() {
	int  passed   = 0;
	int  failures = 0;
	auto report   = [&](bool ok, const std::string& name, const std::string& detail) {
		(ok ? passed : failures)++;
		std::cout << "SHADED-WEATHER: TEST " << (ok ? "PASS " : "FAIL ") << name << " " << detail << std::endl;
	};
	// Room A: walls x, y 4..12 at lifts 0-4, a roof at lift 5 over the
	// inside (5..11), a doorway at (8, 12).  Room B: walls 15..21, roof over
	// 16..20, no way in.
	Shaded_grid_volume vol;
	vol.origin_tx = 1000;
	vol.origin_ty = 2000;
	vol.resize(24, 24, shaded_grid_lifts);
	constexpr uint8_t wall = shaded_cell_solid | shaded_cell_object;
	auto room = [&vol](int lo, int hi) {
		for (int i = lo; i <= hi; i++) {
			vol.stamp_box(i, lo, 0, i, lo, 4, 255, wall);
			vol.stamp_box(i, hi, 0, i, hi, 4, 255, wall);
			vol.stamp_box(lo, i, 0, lo, i, 4, 255, wall);
			vol.stamp_box(hi, i, 0, hi, i, 4, 255, wall);
		}
		vol.stamp_box(lo + 1, lo + 1, 5, hi - 1, hi - 1, 5, 255, wall | shaded_cell_roof);
	};
	room(4, 12);
	room(15, 21);
	for (int z = 0; z <= 4; z++) {
		vol.cells[vol.index(8, 12, z)]     = 0;
		vol.cells[vol.index(8, 12, z) + 1] = 0;
	}
	vol.rebuild_sky(0, 0, vol.size_x, vol.size_y, shaded_grid_sky_threshold);
	constexpr float      fade = 3.0f;
	std::vector<uint8_t> open;
	Shaded_cpu::build_sky_open(vol, fade, open);
	auto at = [&open, &vol](int x, int y) {
		return static_cast<int>(open[static_cast<size_t>(y) * static_cast<size_t>(vol.size_x) + static_cast<size_t>(x)]);
	};
	auto close_to = [](int v, double want) {
		return std::abs(static_cast<double>(v) - want) <= 2.0;
	};
	report(at(2, 2) == 255 && at(8, 13) == 255 && at(8, 12) == 255 && at(23, 0) == 255, "open-outside",
		   "open(2,2)=" + std::to_string(at(2, 2)) + " doorway=" + std::to_string(at(8, 12)));
	report(close_to(at(8, 11), 255.0 - 255.0 / 3.0), "open-doorway-1",
		   "open(8,11)=" + std::to_string(at(8, 11)) + " want 170");
	report(close_to(at(7, 11), 255.0 * (1.0 - 1.41421356 / 3.0)), "open-doorway-diagonal",
		   "open(7,11)=" + std::to_string(at(7, 11)) + " want 135");
	report(at(8, 8) == 0 && at(8, 9) == 0 && at(6, 6) == 0, "open-deep-inside",
		   "open(8,9)=" + std::to_string(at(8, 9)) + " open(8,8)=" + std::to_string(at(8, 8)));
	bool closed = true;
	for (int y = 15; y <= 21; y++) {
		for (int x = 15; x <= 21; x++) {
			closed = closed && at(x, y) == 0;
		}
	}
	report(closed, "open-closed-room", "open(18,18)=" + std::to_string(at(18, 18)));
	// Behind the wall beside the doorway: only the way round through the
	// doorway counts (3.41 tiles: 0), not 2 tiles through the wall (85).
	report(at(5, 11) == 0 && close_to(at(6, 11), 255.0 * (1.0 - 2.41421356 / 3.0)) && at(7, 12) == 0, "open-no-wall-leak",
		   "open(5,11)=" + std::to_string(at(5, 11)) + " open(6,11)=" + std::to_string(at(6, 11)) + " want 0, 50");
	// The per-pixel factor.
	Shaded_weather_set set;
	set.grid    = &vol;
	set.open_w  = vol.size_x;
	set.open_h  = vol.size_y;
	set.open_tx = vol.origin_tx;
	set.open_ty = vol.origin_ty;
	set.open    = open;
	const Shaded_normalf up{0.0f, 0.0f, 1.0f};
	const Shaded_normalf south{0.0f, 1.0f, 0.0f};
	auto o = [&set](float x, float y, float z, const Shaded_normalf& n) {
		return Shaded_weather_open(set, Shaded_point{static_cast<float>(set.open_tx) + x, static_cast<float>(set.open_ty) + y, z}, n);
	};
	const float roof_top = o(8.5f, 8.5f, 6.0f, up);
	const float floor_in = o(8.5f, 6.5f, 0.0f, up);
	const float wall_out = o(6.5f, 13.0f, 2.0f, south);
	const float wall_in  = o(8.5f, 5.0f, 2.0f, south);
	const float ground   = o(2.5f, 2.5f, 0.0f, up);
	report(roof_top == 1.0f, "weather-open-roof-top", "O=" + Fmt(roof_top));
	report(floor_in == 0.0f, "weather-open-floor-inside", "O=" + Fmt(floor_in));
	report(wall_out == 1.0f, "weather-open-south-face-outside", "O=" + Fmt(wall_out));
	report(wall_in == 0.0f, "weather-open-north-wall-inside", "O=" + Fmt(wall_in));
	report(ground == 1.0f, "weather-open-ground", "O=" + Fmt(ground));
	Shaded_weather_set none;
	report(o(8.5f, 6.5f, 0.0f, up) == 0.0f
				   && Shaded_weather_open(none, Shaded_point{1008.5f, 2006.5f, 0.0f}, up) == 1.0f,
		   "weather-open-no-grid", "O(no grid)=" + Fmt(Shaded_weather_open(none, Shaded_point{1008.5f, 2006.5f, 0.0f}, up)));
	// A 16 x 16 square frame round its origin (rows -8..7, plain runs).
	std::vector<unsigned char> rle;
	auto put2 = [&rle](int v) {
		rle.push_back(static_cast<unsigned char>(v & 0xff));
		rle.push_back(static_cast<unsigned char>((v >> 8) & 0xff));
	};
	for (int y = -8; y < 8; y++) {
		put2(16 << 1);
		put2(-8);
		put2(y);
		rle.insert(rle.end(), 16, 5);
	}
	put2(0);
	constexpr int        blur = 3;
	std::vector<uint8_t> data(7, 0);    // Something before it: offsets are honoured.
	Shaded_cloud_mask    mask;
	Shaded_cpu::build_cloud_mask(rle.data(), 8, 8, 16, 16, blur, data, mask);
	auto mval = [&](int x, int y) {
		const int mx = x - mask.x0;
		const int my = y - mask.y0;
		if (mx < 0 || my < 0 || mx >= mask.w || my >= mask.h) {
			return 0;
		}
		return static_cast<int>(data[mask.offset + static_cast<size_t>(my) * static_cast<size_t>(mask.w) + static_cast<size_t>(mx)]);
	};
	const int centre = mval(0, 0);
	const int edge   = mval(7, 0);    // Last painted column.
	const int out    = mval(7 + 2 * blur + 1, 0);
	const int corner = mval(mask.x0, mask.y0);
	bool      mono   = true;
	for (int x = 0; x < mask.x0 + mask.w; x++) {
		mono = mono && mval(x + 1, 0) <= mval(x, 0);
	}
	report(mask.offset == 7 && mask.w == 16 + 4 * blur && mask.x0 == -8 - 2 * blur, "cloud-mask-layout",
		   "offset=" + std::to_string(mask.offset) + " w=" + std::to_string(mask.w) + " x0=" + std::to_string(mask.x0));
	report(centre == 255, "cloud-mask-centre", "centre=" + std::to_string(centre));
	report(out == 0 && corner == 0, "cloud-mask-outside", "outside=" + std::to_string(out) + " corner=" + std::to_string(corner));
	report(edge > 0 && edge < 255, "cloud-mask-edge", "edge=" + std::to_string(edge));
	report(mono, "cloud-mask-monotonic", "row 0 non-increasing towards the edge");
	// The cover lookup: a cloud with its origin at world pixel (800, 1600).
	set.masks.push_back(mask);
	set.mask_data = data;
	set.clouds.push_back(Shaded_cloud{800.0f, 1600.0f, 0, 0});
	const float under = Shaded_cloud_cover(set, Shaded_point{100.0625f, 200.0625f, 0.0f});
	const float away  = Shaded_cloud_cover(set, Shaded_point{110.0f, 200.0f, 0.0f});
	set.clouds.push_back(Shaded_cloud{800.0f, 1600.0f, 0, 0});
	const float twice = Shaded_cloud_cover(set, Shaded_point{100.0625f, 200.0625f, 0.0f});
	report(under == 1.0f && away == 0.0f && twice == 1.0f, "cloud-cover",
		   "under=" + Fmt(under) + " away=" + Fmt(away) + " two clouds=" + Fmt(twice));
	std::cout << "SHADED-WEATHER: self-test " << passed << " passed, " << failures << " failed" << std::endl;
	return failures;
}
