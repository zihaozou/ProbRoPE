#include <metavision/sdk/stream/camera.h>
#include <metavision/hal/facilities/i_trigger_in.h>
#include <metavision/hal/facilities/i_ll_biases.h>
#include <metavision/hal/facilities/i_geometry.h>
#include <metavision/hal/facilities/i_erc_module.h>
#include <metavision/hal/facilities/i_antiflicker_module.h>
#include <metavision/hal/facilities/i_event_trail_filter_module.h>
#include <metavision/hal/facilities/i_roi.h>
#include <metavision/hal/facilities/i_digital_crop.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

extern "C" {

struct MvEventCD { long long t; unsigned short x; unsigned short y; short p; };
struct MvEventTrigger { long long t; short p; short id; };

typedef void (*mv_cd_cb)(const MvEventCD*, size_t, void*);
typedef void (*mv_trig_cb)(const MvEventTrigger*, size_t, void*);
typedef void (*mv_status_cb)(int is_eof, void*);
typedef void (*mv_raw_cb)(const unsigned char*, size_t, void*);

}

namespace {
thread_local std::string g_last_error;

struct MvCam {
    Metavision::Camera cam;
    std::vector<MvEventCD> cd_buf;
    std::vector<MvEventTrigger> trig_buf;
};

template <typename F>
auto guarded(F&& f) -> decltype(f()) {
    try { return f(); }
    catch (const std::exception& e) { g_last_error = e.what(); return decltype(f()){}; }
}

template <typename Facility>
Facility* facility_of(void* h, const char* what) {
    auto* c = static_cast<MvCam*>(h);
    auto* f = c->cam.get_device().get_facility<Facility>();
    if (!f) g_last_error = std::string("no ") + what + " facility";
    return f;
}

using TrailType = Metavision::I_EventTrailFilterModule::Type;

int trail_type_to_int(TrailType t) {
    switch (t) {
        case TrailType::TRAIL: return 0;
        case TrailType::STC_CUT_TRAIL: return 1;
        case TrailType::STC_KEEP_TRAIL: return 2;
    }
    return 0;
}

bool trail_type_from_int(int v, TrailType* out) {
    switch (v) {
        case 0: *out = TrailType::TRAIL; return true;
        case 1: *out = TrailType::STC_CUT_TRAIL; return true;
        case 2: *out = TrailType::STC_KEEP_TRAIL; return true;
    }
    g_last_error = "invalid trail filter type (want 0..2)";
    return false;
}
}

extern "C" {

const char* mv_last_error() { return g_last_error.c_str(); }

void* mv_open_live() {
    return guarded([]() -> void* {
        auto* c = new MvCam{Metavision::Camera::from_first_available()};
        return c;
    });
}

void* mv_open_file(const char* path, int realtime) {
    return guarded([=]() -> void* {
        Metavision::FileConfigHints hints;
        hints.real_time_playback(realtime != 0);
        auto* c = new MvCam{Metavision::Camera::from_file(path, hints)};
        return c;
    });
}

int mv_get_geometry(void* h, int* w, int* hgt) {
    auto* c = static_cast<MvCam*>(h);
    return guarded([=]() -> int {
        auto& g = c->cam.get_facility<Metavision::I_Geometry>();
        *w = (int)g.get_width();
        *hgt = (int)g.get_height();
        return 1;
    });
}

int mv_set_cd_callback(void* h, mv_cd_cb cb, void* user) {
    auto* c = static_cast<MvCam*>(h);
    return guarded([=]() -> int {
        c->cam.cd().add_callback([=](const Metavision::EventCD* b, const Metavision::EventCD* e) {
            auto* mc = static_cast<MvCam*>(h);
            mc->cd_buf.clear();
            mc->cd_buf.reserve(e - b);
            for (auto* ev = b; ev != e; ++ev)
                mc->cd_buf.push_back({(long long)ev->t, ev->x, ev->y, ev->p});
            cb(mc->cd_buf.data(), mc->cd_buf.size(), user);
        });
        return 1;
    });
}

int mv_set_trigger_callback(void* h, mv_trig_cb cb, void* user) {
    auto* c = static_cast<MvCam*>(h);
    return guarded([=]() -> int {
        c->cam.ext_trigger().add_callback(
            [=](const Metavision::EventExtTrigger* b, const Metavision::EventExtTrigger* e) {
                auto* mc = static_cast<MvCam*>(h);
                mc->trig_buf.clear();
                for (auto* ev = b; ev != e; ++ev)
                    mc->trig_buf.push_back({(long long)ev->t, ev->p, ev->id});
                cb(mc->trig_buf.data(), mc->trig_buf.size(), user);
            });
        return 1;
    });
}

int mv_set_raw_callback(void* h, mv_raw_cb cb, void* user) {
    auto* c = static_cast<MvCam*>(h);
    return guarded([=]() -> int {
        c->cam.raw_data().add_callback([=](const std::uint8_t* data, size_t size) {
            cb(data, size, user);
        });
        return 1;
    });
}

int mv_set_status_callback(void* h, mv_status_cb cb, void* user) {
    auto* c = static_cast<MvCam*>(h);
    return guarded([=]() -> int {
        c->cam.add_status_change_callback([=](const Metavision::CameraStatus& s) {
            cb(s == Metavision::CameraStatus::STOPPED ? 1 : 0, user);
        });
        return 1;
    });
}

int mv_enable_trigger_in(void* h) {
    auto* c = static_cast<MvCam*>(h);
    return guarded([=]() -> int {
        auto* t = c->cam.get_device().get_facility<Metavision::I_TriggerIn>();
        if (!t) { g_last_error = "no I_TriggerIn facility (file replay?)"; return 0; }
        return t->enable(Metavision::I_TriggerIn::Channel::Main) ? 1 : 0;
    });
}

int mv_set_bias(void* h, const char* name, int value) {
    auto* c = static_cast<MvCam*>(h);
    return guarded([=]() -> int {
        auto* b = c->cam.get_device().get_facility<Metavision::I_LL_Biases>();
        if (!b) { g_last_error = "no I_LL_Biases facility"; return 0; }
        return b->set(name, value) ? 1 : 0;
    });
}

int mv_get_bias(void* h, const char* name, int* value) {
    auto* c = static_cast<MvCam*>(h);
    return guarded([=]() -> int {
        auto* b = c->cam.get_device().get_facility<Metavision::I_LL_Biases>();
        if (!b) { g_last_error = "no I_LL_Biases facility"; return 0; }
        *value = b->get(name);
        return 1;
    });
}


int mv_probe_facilities(void* h, int* has_biases, int* has_erc, int* has_antiflicker,
                         int* has_trail_filter, int* has_roi, int* has_digital_crop) {
    auto* c = static_cast<MvCam*>(h);
    return guarded([=]() -> int {
        auto& dev = c->cam.get_device();
        *has_biases = dev.get_facility<Metavision::I_LL_Biases>() != nullptr;
        *has_erc = dev.get_facility<Metavision::I_ErcModule>() != nullptr;
        *has_antiflicker = dev.get_facility<Metavision::I_AntiFlickerModule>() != nullptr;
        *has_trail_filter = dev.get_facility<Metavision::I_EventTrailFilterModule>() != nullptr;
        *has_roi = dev.get_facility<Metavision::I_ROI>() != nullptr;
        *has_digital_crop = dev.get_facility<Metavision::I_DigitalCrop>() != nullptr;
        return 1;
    });
}

int mv_get_bias_info(void* h, const char* name, int* min_rec, int* max_rec, int* min_alw, int* max_alw) {
    auto* c = static_cast<MvCam*>(h);
    return guarded([=]() -> int {
        auto* b = c->cam.get_device().get_facility<Metavision::I_LL_Biases>();
        if (!b) { g_last_error = "no I_LL_Biases facility"; return 0; }
        Metavision::LL_Bias_Info info;
        if (!b->get_bias_info(name, info)) { g_last_error = "get_bias_info failed"; return 0; }
        auto rec = info.get_bias_recommended_range();
        auto alw = info.get_bias_allowed_range();
        *min_rec = rec.first;
        *max_rec = rec.second;
        *min_alw = alw.first;
        *max_alw = alw.second;
        return 1;
    });
}



int mv_erc_enable(void* h, int enable) {
    return guarded([=]() -> int {
        auto* e = facility_of<Metavision::I_ErcModule>(h, "I_ErcModule");
        if (!e) return 0;
        return e->enable(enable != 0) ? 1 : 0;
    });
}

int mv_erc_is_enabled(void* h, int* enabled) {
    return guarded([=]() -> int {
        auto* e = facility_of<Metavision::I_ErcModule>(h, "I_ErcModule");
        if (!e) return 0;
        *enabled = e->is_enabled() ? 1 : 0;
        return 1;
    });
}

int mv_erc_set_cd_event_count(void* h, uint32_t event_count) {
    return guarded([=]() -> int {
        auto* e = facility_of<Metavision::I_ErcModule>(h, "I_ErcModule");
        if (!e) return 0;
        return e->set_cd_event_count(event_count) ? 1 : 0;
    });
}

int mv_erc_get_cd_event_count(void* h, uint32_t* event_count) {
    return guarded([=]() -> int {
        auto* e = facility_of<Metavision::I_ErcModule>(h, "I_ErcModule");
        if (!e) return 0;
        *event_count = e->get_cd_event_count();
        return 1;
    });
}

int mv_erc_get_min_max_cd_event_count(void* h, uint32_t* min, uint32_t* max) {
    return guarded([=]() -> int {
        auto* e = facility_of<Metavision::I_ErcModule>(h, "I_ErcModule");
        if (!e) return 0;
        *min = e->get_min_supported_cd_event_count();
        *max = e->get_max_supported_cd_event_count();
        return 1;
    });
}

int mv_erc_get_count_period(void* h, uint32_t* period_us) {
    return guarded([=]() -> int {
        auto* e = facility_of<Metavision::I_ErcModule>(h, "I_ErcModule");
        if (!e) return 0;
        *period_us = e->get_count_period();
        return 1;
    });
}


int mv_af_enable(void* h, int enable) {
    return guarded([=]() -> int {
        auto* a = facility_of<Metavision::I_AntiFlickerModule>(h, "I_AntiFlickerModule");
        if (!a) return 0;
        return a->enable(enable != 0) ? 1 : 0;
    });
}

int mv_af_is_enabled(void* h, int* enabled) {
    return guarded([=]() -> int {
        auto* a = facility_of<Metavision::I_AntiFlickerModule>(h, "I_AntiFlickerModule");
        if (!a) return 0;
        *enabled = a->is_enabled() ? 1 : 0;
        return 1;
    });
}

int mv_af_set_frequency_band(void* h, uint32_t low_hz, uint32_t high_hz) {
    return guarded([=]() -> int {
        auto* a = facility_of<Metavision::I_AntiFlickerModule>(h, "I_AntiFlickerModule");
        if (!a) return 0;
        return a->set_frequency_band(low_hz, high_hz) ? 1 : 0;
    });
}

int mv_af_get_frequency_band(void* h, uint32_t* low_hz, uint32_t* high_hz) {
    return guarded([=]() -> int {
        auto* a = facility_of<Metavision::I_AntiFlickerModule>(h, "I_AntiFlickerModule");
        if (!a) return 0;
        *low_hz = a->get_band_low_frequency();
        *high_hz = a->get_band_high_frequency();
        return 1;
    });
}

int mv_af_get_supported_frequency_range(void* h, uint32_t* min_hz, uint32_t* max_hz) {
    return guarded([=]() -> int {
        auto* a = facility_of<Metavision::I_AntiFlickerModule>(h, "I_AntiFlickerModule");
        if (!a) return 0;
        *min_hz = a->get_min_supported_frequency();
        *max_hz = a->get_max_supported_frequency();
        return 1;
    });
}

int mv_af_set_mode(void* h, int mode) {
    return guarded([=]() -> int {
        auto* a = facility_of<Metavision::I_AntiFlickerModule>(h, "I_AntiFlickerModule");
        if (!a) return 0;
        auto m = mode == 0 ? Metavision::I_AntiFlickerModule::BAND_PASS
                           : Metavision::I_AntiFlickerModule::BAND_STOP;
        return a->set_filtering_mode(m) ? 1 : 0;
    });
}

int mv_af_get_mode(void* h, int* mode) {
    return guarded([=]() -> int {
        auto* a = facility_of<Metavision::I_AntiFlickerModule>(h, "I_AntiFlickerModule");
        if (!a) return 0;
        *mode = a->get_filtering_mode() == Metavision::I_AntiFlickerModule::BAND_PASS ? 0 : 1;
        return 1;
    });
}


int mv_trail_enable(void* h, int enable) {
    return guarded([=]() -> int {
        auto* t = facility_of<Metavision::I_EventTrailFilterModule>(h, "I_EventTrailFilterModule");
        if (!t) return 0;
        return t->enable(enable != 0) ? 1 : 0;
    });
}

int mv_trail_is_enabled(void* h, int* enabled) {
    return guarded([=]() -> int {
        auto* t = facility_of<Metavision::I_EventTrailFilterModule>(h, "I_EventTrailFilterModule");
        if (!t) return 0;
        *enabled = t->is_enabled() ? 1 : 0;
        return 1;
    });
}

int mv_trail_set_type(void* h, int type) {
    return guarded([=]() -> int {
        auto* t = facility_of<Metavision::I_EventTrailFilterModule>(h, "I_EventTrailFilterModule");
        if (!t) return 0;
        TrailType tt;
        if (!trail_type_from_int(type, &tt)) return 0;
        return t->set_type(tt) ? 1 : 0;
    });
}

int mv_trail_get_type(void* h, int* type) {
    return guarded([=]() -> int {
        auto* t = facility_of<Metavision::I_EventTrailFilterModule>(h, "I_EventTrailFilterModule");
        if (!t) return 0;
        *type = trail_type_to_int(t->get_type());
        return 1;
    });
}

int mv_trail_get_available_types(void* h, int* type_bitmask) {
    return guarded([=]() -> int {
        auto* t = facility_of<Metavision::I_EventTrailFilterModule>(h, "I_EventTrailFilterModule");
        if (!t) return 0;
        int mask = 0;
        for (auto ty : t->get_available_types())
            mask |= 1 << trail_type_to_int(ty);
        *type_bitmask = mask;
        return 1;
    });
}

int mv_trail_set_threshold(void* h, uint32_t threshold_us) {
    return guarded([=]() -> int {
        auto* t = facility_of<Metavision::I_EventTrailFilterModule>(h, "I_EventTrailFilterModule");
        if (!t) return 0;
        return t->set_threshold(threshold_us) ? 1 : 0;
    });
}

int mv_trail_get_threshold(void* h, uint32_t* threshold_us) {
    return guarded([=]() -> int {
        auto* t = facility_of<Metavision::I_EventTrailFilterModule>(h, "I_EventTrailFilterModule");
        if (!t) return 0;
        *threshold_us = t->get_threshold();
        return 1;
    });
}

int mv_trail_get_threshold_range(void* h, uint32_t* min_us, uint32_t* max_us) {
    return guarded([=]() -> int {
        auto* t = facility_of<Metavision::I_EventTrailFilterModule>(h, "I_EventTrailFilterModule");
        if (!t) return 0;
        *min_us = t->get_min_supported_threshold();
        *max_us = t->get_max_supported_threshold();
        return 1;
    });
}

int mv_start(void* h) {
    auto* c = static_cast<MvCam*>(h);
    return guarded([=]() -> int { return c->cam.start() ? 1 : 0; });
}

int mv_stop(void* h) {
    auto* c = static_cast<MvCam*>(h);
    return guarded([=]() -> int { return c->cam.stop() ? 1 : 0; });
}

int mv_start_recording(void* h, const char* path) {
    auto* c = static_cast<MvCam*>(h);
    return guarded([=]() -> int { return c->cam.start_recording(path) ? 1 : 0; });
}

int mv_stop_recording(void* h) {
    auto* c = static_cast<MvCam*>(h);
    return guarded([=]() -> int { return c->cam.stop_recording() ? 1 : 0; });
}

void mv_close(void* h) { delete static_cast<MvCam*>(h); }

}
