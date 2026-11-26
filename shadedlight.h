/*
 *  shadedlight.h - Height-aware "shaded light" scene record.
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

/*
 *  Stage 1 of an alternative lighting renderer: while the software painter
 *  draws the world, record for every pixel it writes the height of the
 *  surface, the face it lies on and which object drew it.  Later stages
 *  light the frame from this record.
 *
 *  The buffers are plain arrays with one fixed-size element per pixel of
 *  the 8-bit draw surface (guard band included, same pitch), so they can be
 *  uploaded as textures unchanged.  Every step is a function over those
 *  buffers behind a backend switch; only the CPU backend exists so far.
 *  The tunables live in Shaded_light_params, laid out so that it can be
 *  passed to a shader as a uniform block (std140) unchanged.
 */

#ifndef SHADEDLIGHT_H
#define SHADEDLIGHT_H 1

#include "shapeid.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

class Game_object;
class Image_buffer8;
class Image_window8;

// Facing values: the low 3 bits of each facing pixel.
constexpr uint8_t shaded_face_none    = 0;
constexpr uint8_t shaded_face_ground  = 1;    // Flats, terrain, 0-height objects.
constexpr uint8_t shaded_face_top     = 2;
constexpr uint8_t shaded_face_south   = 3;    // Faces +y.
constexpr uint8_t shaded_face_east    = 4;    // Faces +x.
constexpr uint8_t shaded_face_rounded = 5;    // Reserved for rounded volumes.
constexpr uint8_t shaded_face_mask    = 0x07;
// Set when the pixel lay outside the object's projected 3D box and its face
// and height were clamped onto the box.
constexpr uint8_t shaded_face_clamped = 0x08;

// Height units per lift in the height buffer (one screen pixel = 1/4 lift).
constexpr int shaded_height_units_per_lift = 16;
// Object ids: 0 = no object, 1..65534 = 1 + index into the object table.
constexpr uint16_t shaded_id_none     = 0;
constexpr uint16_t shaded_id_overflow = 0xffff;

enum class Shaded_view : int32_t {
	normal = 0,
	height,
	facing,
	object_id,
	count
};

enum class Shaded_backend : int32_t {
	cpu = 0,
	gpu = 1    // Not built yet (needs the SDL GPU device); falls back to CPU.
};

// How an object's pixels get their geometry.
enum class Shaded_volume : uint8_t {
	box,       // The object's 3D box (Shape_info dimensions).
	rounded    // People, trees and so on; a later stage.  Treated as box now.
};

/*
 *  All tunables, in vec4-sized groups of plain 32-bit values (std140).
 */
struct alignas(16) Shaded_light_params {
	int32_t enabled      = 0;
	int32_t view         = 0;    // Shaded_view.
	int32_t backend      = 0;    // Shaded_backend.
	int32_t object_count = 0;
	// Buffer geometry: element (x, y) of the game area is at
	//   buffer_origin + y * buffer_stride + x.
	int32_t buffer_width  = 0;
	int32_t buffer_height = 0;
	int32_t buffer_stride = 0;
	int32_t buffer_origin = 0;
	// Scroll position of the recorded frame, to rebuild world positions.
	int32_t scroll_tx = 0;    // Top-left tile.
	int32_t scroll_ty = 0;
	int32_t scroll_px = 0;    // Extra pixel offset (smooth scrolling).
	int32_t scroll_py = 0;
	float   pixels_per_tile       = 8.0f;
	float   pixels_per_lift       = 4.0f;
	float   height_units_per_lift = static_cast<float>(shaded_height_units_per_lift);
	float   debug_height_range    = 16.0f;    // Lifts covered by the height ramp.
	// Global sliders for stage 2 (unused so far).
	float ambient      = 1.0f;
	float brightness   = 1.0f;
	float radius_scale = 1.0f;
	float reserved0    = 0.0f;
};

static_assert(sizeof(Shaded_light_params) % 16 == 0);
static_assert(std::is_trivially_copyable_v<Shaded_light_params> && std::is_standard_layout_v<Shaded_light_params>);

/*
 *  One painted object, as uploaded to a storage buffer later.
 */
struct Shaded_object_record {
	int32_t tx, ty, tz;                // Tile (SE bottom corner of the box).
	int32_t xtiles, ytiles, ztiles;    // 3D box (ztiles in lifts).
	int32_t shape, frame;
};

static_assert(sizeof(Shaded_object_record) == 32);

/*
 *  A rectangle in game-area pixels.
 */
struct Shaded_rect {
	int x = 0, y = 0, w = 0, h = 0;
};

/*
 *  The scene record.
 */
struct Shaded_scene_buffers {
	int                               width  = 0;    // Of the draw surface.
	int                               height = 0;
	int                               stride = 0;    // Elements per row.
	int                               origin = 0;    // Index of game pixel (0, 0).
	std::vector<uint16_t>             height_map;    // 1/16 lift above lift 0.
	std::vector<uint8_t>              facing_map;    // shaded_face_* (+ clamped bit).
	std::vector<uint16_t>             object_ids;    // shaded_id_*.
	std::vector<Shaded_object_record> objects;
	// Game-area rectangle the buffers cover.
	Shaded_rect bounds() const;

	int index(int x, int y) const {
		return origin + y * stride + x;
	}
};

/*
 *  Where a pixel lands on an object's box.
 */
struct Shaded_box_hit {
	uint8_t facing;    // Face, maybe with shaded_face_clamped.
	int     dz_px;     // Height above the box bottom, in pixels (0..hpx).
};

/*
 *  Classify a pixel against a box projected by Exult's view: one tile is
 *  8 px, one lift moves 4 px up and 4 px left.  (u, v) is the pixel minus
 *  the box's origin pixel (its SE bottom corner, from get_shape_location),
 *  so the box footprint is u in (-xpx, 0], v in (-ypx, 0].  xpx, ypx and
 *  hpx are the box size in pixels (xtiles * 8, ytiles * 8, ztiles * 4).
 *  The view ray moves +1 px in world x, y and height per pixel of depth, so
 *  the top is met first, then the east face (u >= v) or the south face.
 *  Mirrored (bit 5) frames need nothing extra: Shape_info already swaps
 *  their x and y sizes, which swaps the faces.
 */
constexpr Shaded_box_hit Shaded_classify_box(int u, int v, int xpx, int ypx, int hpx) {
	auto in_range = [](int val, int size) {
		return val > -size && val <= 0;
	};
	if (hpx <= 0) {    // Flat: everything is ground.
		const bool inside = in_range(u, xpx) && in_range(v, ypx);
		return {static_cast<uint8_t>(shaded_face_ground | (inside ? 0 : shaded_face_clamped)), 0};
	}
	if (in_range(u + hpx, xpx) && in_range(v + hpx, ypx)) {
		return {shaded_face_top, hpx};
	}
	const bool    east   = u >= v;
	const uint8_t face   = east ? shaded_face_east : shaded_face_south;
	int           dz     = east ? -u : -v;
	bool          inside = east ? in_range(v - u, ypx) : in_range(u - v, xpx);
	if (dz < 0 || dz > hpx) {
		inside = false;
		dz     = dz < 0 ? 0 : hpx;
	}
	return {static_cast<uint8_t>(face | (inside ? 0 : shaded_face_clamped)), dz};
}

/*
 *  What the per-pixel steps need to know about the object being painted.
 */
struct Shaded_object_context {
	int           ox = 0, oy = 0;    // Origin pixel of the box.
	int           xpx = 0, ypx = 0, hpx = 0;
	int           base_height = 0;    // Box bottom, in height units.
	uint16_t      id          = shaded_id_none;
	Shaded_volume volume      = Shaded_volume::box;
};

/*
 *  CPU backend: each step is a plain function over the buffers.  Rectangles
 *  must already be clipped to the buffers' bounds.
 */
namespace Shaded_cpu {
	// Reset a rectangle to facing none, height 0, id 0.
	void clear_rect(Shaded_scene_buffers& buf, const Shaded_rect& r);
	// Fill a rectangle with one flat plane.
	void fill_plane(Shaded_scene_buffers& buf, const Shaded_rect& r, uint16_t height, uint8_t facing, uint16_t id);
	// Record the pixels an RLE shape paints at (xoff, yoff), clipped to clip.
	// Pixel values in [skip_from, 0xfe] are skipped (translucent).
	void record_spans(
			Shaded_scene_buffers& buf, const Shaded_rect& clip, const Shaded_object_context& obj, const unsigned char* rle,
			int xoff, int yoff, int skip_from);
	// Overwrite a rectangle of an 8-bit frame (indexed like the buffers) with
	// the debug view; lut maps a false colour code to a palette index.
	void resolve_view(
			const Shaded_scene_buffers& buf, const Shaded_light_params& params, const Shaded_rect& r, const uint8_t* lut,
			unsigned char* frame);
	// False colour code (index into the lut) for one pixel of a view.
	int view_code(const Shaded_scene_buffers& buf, const Shaded_light_params& params, int index);
}    // namespace Shaded_cpu

// Which volume an object uses.  Box for everything in stage 1; people,
// trees and other organic shapes become rounded in a later stage.
Shaded_volume Shaded_volume_for(const Game_object& obj);

/*
 *  The scene recorder.  One instance (get()); recording() is non-null only
 *  while the world is being painted with recording on.
 */
class Shaded_light final : public Shape_paint_listener {
public:
	static Shaded_light& get();

	static Shaded_light* recording() {
		return active;
	}

	bool is_enabled() const {
		return record_on;
	}

	// Around Game_render::paint_map in Game_window::paint.
	void begin_paint(Image_window8& win, bool full);
	void end_paint(Image_window8& win, bool full);
	// Around obj->paint() for a world object (ox, oy = its shape location).
	void begin_object(const Game_object& obj, int ox, int oy);
	void end_object();
	// Record a flat plane (chunk flats, dungeon blackness) at a lift.
	void record_plane(int x, int y, int w, int h, int lift, uint8_t face);
	// Next debug view; returns it.
	Shaded_view cycle_view();
	static const char* view_name(Shaded_view view);
	// World position (tiles, lifts) of a recorded game pixel.
	void world_position(int x, int y, float& wx, float& wy, float& wz) const;

	const Shaded_light_params& get_params() const {
		return params;
	}

	const Shaded_scene_buffers& get_buffers() const {
		return buffers;
	}

	void shape_painted(int xoff, int yoff, Shape_frame& shape, bool translucent, int xfcnt) override;
	~Shaded_light() override;

private:
	using Clock = std::chrono::steady_clock;

	static Shaded_light*  active;    // Set while recording a paint.
	Shaded_light_params   params;
	Shaded_scene_buffers  buffers;
	Image_buffer8*        target = nullptr;    // Buffer painted into.
	Shaded_rect           clip;                // Paint clip, within bounds.
	Shaded_object_context object;
	bool                  in_object = false;
	bool                  record_on = false;    // Config or dump.
	std::unordered_map<const Game_object*, uint16_t> object_index;
	uint16_t                                         main_actor_id = shaded_id_none;
	// Headless dumps (EXULT_SHADED_DUMP).
	std::string      dump_dir;
	std::vector<int> dump_at;    // ms after the first recorded full paint.
	size_t           dump_next = 0;
	bool             clock_started = false;
	Clock::time_point first_paint;
	// Timing (EXULT_SHADED_TIMING).
	bool              timing = false;
	Clock::time_point paint_start;
	long long         full_paints     = 0;
	double            paint_map_us    = 0;    // Totals over full paints.
	double            record_us       = 0;
	double            paint_record_us = 0;    // In the current paint.
	long long         paints          = 0;    // All paints, full or not.
	double            paint_pixels    = 0;    // Total clip area.
	double            all_paint_us    = 0;
	double            all_record_us   = 0;
	int               game_width      = 0;    // Of the last full paint.
	int               game_height     = 0;
	bool              warned_backend  = false;

	Shaded_light();
	Shaded_backend backend();
	bool           setup_buffers(Image_window8& win);
	void           shift(int dx, int dy);
	void           build_lut(uint8_t* lut) const;
	void           dump(Image_window8& win, int number, bool view, bool full);
	void           report_timing() const;
};

#endif
