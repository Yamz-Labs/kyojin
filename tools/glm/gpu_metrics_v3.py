"""Decode amdgpu gpu_metrics v3.0 (APU, Strix Halo): clocks, power, temps and cumulative throttle residencies."""
import ctypes, sys

u16, u32, u64 = ctypes.c_uint16, ctypes.c_uint32, ctypes.c_uint64
class GM3(ctypes.Structure):
    _fields_ = [("structure_size", u16), ("format_revision", ctypes.c_uint8), ("content_revision", ctypes.c_uint8),
                ("temperature_gfx", u16), ("temperature_soc", u16), ("temperature_core", u16 * 16), ("temperature_skin", u16),
                ("average_gfx_activity", u16), ("average_vcn_activity", u16), ("average_ipu_activity", u16 * 8),
                ("average_core_c0_activity", u16 * 16), ("average_dram_reads", u16), ("average_dram_writes", u16),
                ("average_ipu_reads", u16), ("average_ipu_writes", u16), ("system_clock_counter", u64),
                ("average_socket_power", u32), ("average_ipu_power", u16), ("average_apu_power", u32),
                ("average_gfx_power", u32), ("average_dgpu_power", u32), ("average_all_core_power", u32),
                ("average_core_power", u16 * 16), ("average_sys_power", u16), ("stapm_power_limit", u16),
                ("current_stapm_power_limit", u16), ("average_gfxclk_frequency", u16), ("average_socclk_frequency", u16),
                ("average_vpeclk_frequency", u16), ("average_ipuclk_frequency", u16), ("average_fclk_frequency", u16),
                ("average_vclk_frequency", u16), ("average_uclk_frequency", u16), ("average_mpipu_frequency", u16),
                ("current_coreclk", u16 * 16), ("current_core_maxfreq", u16), ("current_gfx_maxfreq", u16),
                ("throttle_residency_prochot", u32), ("throttle_residency_spl", u32), ("throttle_residency_fppt", u32),
                ("throttle_residency_sppt", u32), ("throttle_residency_thm_core", u32), ("throttle_residency_thm_gfx", u32),
                ("throttle_residency_thm_soc", u32), ("time_filter_alphavalue", u32)]
KEYS = ["temperature_gfx", "temperature_soc", "average_gfx_activity", "average_dram_reads", "average_dram_writes",
        "average_socket_power", "average_gfx_power", "average_all_core_power", "stapm_power_limit", "current_stapm_power_limit",
        "average_gfxclk_frequency", "average_fclk_frequency", "average_uclk_frequency", "average_socclk_frequency",
        "current_gfx_maxfreq", "throttle_residency_prochot", "throttle_residency_spl", "throttle_residency_fppt",
        "throttle_residency_sppt", "throttle_residency_thm_core", "throttle_residency_thm_gfx", "throttle_residency_thm_soc"]

def read(path="/sys/class/drm/card1/device/gpu_metrics"):
    b = open(path, "rb").read()
    g = GM3.from_buffer_copy(b[:ctypes.sizeof(GM3)].ljust(ctypes.sizeof(GM3), b"\0"))
    return {k: getattr(g, k) for k in KEYS}, len(b), ctypes.sizeof(GM3)

if __name__ == "__main__":
    d, n, s = read()
    print("file", n, "struct", s); print(d)
