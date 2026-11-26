// Table-address adapter for Cycles' unmodified lookup routines. The table
// contents come directly from scene/shader.tables in the pinned submodule.
struct ClosureTables {
  int ggx_E = 0 << 16, ggx_Eavg = 1 << 16;
  int ggx_glass_E = 2 << 16, ggx_glass_Eavg = 3 << 16;
  int ggx_glass_inv_E = 4 << 16, ggx_glass_inv_Eavg = 5 << 16;
  int sheen_ltc = 6 << 16, ggx_gen_schlick_ior_s = 7 << 16;
  int ggx_gen_schlick_s = 8 << 16, thin_film_table = 9 << 16;
};
struct ClosureKernelData { ClosureTables tables; };
#define kernel_data ClosureKernelData{}

VN_INLINE float closure_table_fetch(int address) {
  int index = address & 65535;
  switch (address >> 16) {
    case 0: return table_ggx_E[index];
    case 1: return table_ggx_Eavg[index];
    case 2: return table_ggx_glass_E[index];
    case 3: return table_ggx_glass_Eavg[index];
    case 4: return table_ggx_glass_inv_E[index];
    case 5: return table_ggx_glass_inv_Eavg[index];
    case 6: return table_sheen_ltc[index];
    case 7: return table_ggx_gen_schlick_ior_s[index];
    case 8: return table_ggx_gen_schlick_s[index];
    default: {
      // ShaderManager::compute_thin_film_table: XYZ -> scene-linear Rec.709,
      // normalized by the real DC component. Real/imaginary channels use
      // the same normalization. No runtime Blender color-management state.
      int channel = index / 512, row = index % 512;
      float3 m = channel % 3 == 0 ? make_float3(3.2404542f, -1.5371385f, -0.4985314f)
               : channel % 3 == 1 ? make_float3(-0.9692660f, 1.8760108f, 0.0415560f)
                                  : make_float3(0.0556434f, -0.2040259f, 1.0572252f);
      const float *value = table_thin_film_cmf[row] + (channel >= 3 ? 3 : 0);
      const float *dc = table_thin_film_cmf[0];
      return dot(m, make_float3(value[0], value[1], value[2])) /
             dot(m, make_float3(dc[0], dc[1], dc[2]));
    }
  }
}
#define kernel_data_fetch(name, index) closure_table_fetch(index)
