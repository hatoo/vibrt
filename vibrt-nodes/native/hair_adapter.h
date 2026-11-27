// The pinned legacy reflection sampler reports cos(phi/2), whereas its
// inverse-CDF uses phi/roughness2. Re-evaluate the sampled direction with the
// upstream evaluator so its PDF matches the actual distribution and NEE MIS.
// This also rejects samples outside the closure's reflection/transmission side.
VN_INLINE int sample_classic_hair(const HairBsdf *hair, bool transmission, float3 Ng,
                                  float3 incoming, float2 random, Spectrum *value,
                                  float3 *direction, float *pdf, float2 *roughness) {
  auto sc = reinterpret_cast<const ShaderClosure *>(hair);
  int label = transmission ? bsdf_hair_transmission_sample(sc, Ng, incoming, random, value,
                                                           direction, pdf, roughness)
                           : bsdf_hair_reflection_sample(sc, Ng, incoming, random, value, direction,
                                                         pdf, roughness);
  if (*pdf > 0) {
    *value = transmission ? bsdf_hair_transmission_eval(sc, incoming, *direction, pdf)
                          : bsdf_hair_reflection_eval(sc, incoming, *direction, pdf);
  }
  return label;
}
