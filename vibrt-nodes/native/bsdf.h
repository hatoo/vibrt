// Renderer-independent assembly of Cycles surface closures. Numerical setup,
// evaluation, sampling and PDFs are the upstream functions in cycles_nodes.h.
// Layer ordering follows kernel/svm/closure.h; allocation and socket access are
// adapted to explicit values instead of Blender's ShaderData/SVM stack.
#pragma once
#include "cycles_nodes.h"

namespace vibrt_surface {
using namespace vibrt_cycles;
using vibrt_cycles::float2;
using vibrt_cycles::float3;
using vibrt_cycles::make_float2;
using vibrt_cycles::make_float3;
enum Model { PRINCIPLED, DIFFUSE, GLOSSY, GLASS, REFRACTION, SHEEN, METALLIC, CONDUCTOR, VELVET };
enum Distribution { GGX, BECKMANN, MULTI_GGX };
struct Parameters {
  float3 N, Ng, T, wi, coat_normal;
  float3 color, specular_tint, coat_tint, sheen_tint, conductor_ior, conductor_k;
  float alpha_x, alpha_y, alpha, roughness, ior, metallic, transmission;
  float specular_level, coat, coat_alpha, coat_ior, sheen, sheen_roughness;
  float thin_film_thickness, thin_film_ior, subsurface, translucent, diffuse_roughness;
  int model, distribution;
  bool backfacing;
};
struct Closures {
  ShaderData sd;
  OrenNayarBsdf diffuse;
  DiffuseBsdf translucent;
  SheenBsdf sheen;
  VelvetBsdf velvet;
  MicrofacetBsdf micro[4]; // coat, metal, transmission, dielectric
  FresnelF82Tint metal_fresnel;
  FresnelConductor conductor_fresnel;
  FresnelGeneralizedSchlick glass_fresnel, specular_fresnel;
  ShaderClosure *entries[7];
  int count;
  float total;
  float3 emission_weight, subsurface_weight;
};
template<class T> VN_INLINE ShaderClosure *closure(T &v) {
  return reinterpret_cast<ShaderClosure *>(&v);
}
template<class T> VN_INLINE void initialize(T &v, float3 normal, float3 weight) {
  v.N = normal;
  v.weight = weight;
  v.sample_weight = fabsf(average(weight));
}
template<class T> VN_INLINE void add(Closures &c, T &v) {
  if (v.sample_weight > 0 && !is_zero(v.weight) && v.type != CLOSURE_NONE_ID) {
    c.entries[c.count++] = closure(v);
    c.total += v.sample_weight;
  }
}
VN_INLINE void setup_micro(MicrofacetBsdf &b, const Parameters &p, float3 n,
                           float3 weight, float ax, float ay, float ior, int kind) {
  initialize(b, n, weight);
  b.T = p.T;
  b.alpha_x = ax; b.alpha_y = ay; b.ior = ior;
  if (p.distribution == BECKMANN) {
    if (kind == GLASS) bsdf_microfacet_beckmann_glass_setup(&b);
    else if (kind == REFRACTION) bsdf_microfacet_beckmann_refraction_setup(&b);
    else bsdf_microfacet_beckmann_setup(&b);
  } else {
    if (kind == GLASS) bsdf_microfacet_ggx_glass_setup(&b);
    else if (kind == REFRACTION) bsdf_microfacet_ggx_refraction_setup(&b);
    else bsdf_microfacet_ggx_setup(&b);
  }
}
VN_INLINE float3 albedo(Closures &c, MicrofacetBsdf &b) {
  return b.weight * bsdf_microfacet_estimate_albedo(0, &c.sd, &b, true, false);
}
VN_INLINE FresnelThinFilm film(const Parameters &p, float ior) {
  return {fmaxf(p.thin_film_thickness, 0), p.backfacing ? p.thin_film_ior / ior : p.thin_film_ior};
}
VN_INLINE void generalized(Closures &c, MicrofacetBsdf &b, FresnelGeneralizedSchlick &f,
                           const Parameters &p, float3 f0, float3 transmission,
                           float3 reflection = one_spectrum(), float exponent_ior = 0) {
  f.thin_film = film(p, p.ior);
  if (p.model == PRINCIPLED && is_zero(transmission)) f.thin_film.ior = p.thin_film_ior;
  f.f0 = f0; f.f90 = one_spectrum(); f.exponent = -(exponent_ior > 0 ? exponent_ior : b.ior);
  f.reflection_tint = reflection; f.transmission_tint = transmission;
  bsdf_microfacet_setup_fresnel_generalized_schlick(0, &b, &c.sd, &f,
                                                 p.distribution == MULTI_GGX);
}
VN_INLINE void setup_layers(const Parameters &p, Closures &c, float3 &weight) {
  if (p.sheen > CLOSURE_WEIGHT_CUTOFF) {
    initialize(c.sheen, safe_normalize(mix(p.N, p.coat_normal, saturatef(p.coat))),
               p.sheen * max(p.sheen_tint, zero_spectrum()) * weight);
    c.sheen.roughness = p.sheen_roughness;
    if (bsdf_sheen_setup(0, &c.sd, &c.sheen)) {
      weight = closure_layering_weight(c.sheen.weight, weight);
      add(c, c.sheen);
    }
  }
  if (p.coat > CLOSURE_WEIGHT_CUTOFF) {
    auto &b = c.micro[0];
    float3 n = maybe_ensure_valid_specular_reflection(&c.sd, p.coat_normal);
    // Principled coat is always GGX, independent of the base distribution.
    initialize(b, n, p.coat * weight);
    b.T = zero_float3(); b.alpha_x = b.alpha_y = p.coat_alpha; b.ior = p.coat_ior;
    bsdf_microfacet_ggx_setup(&b);
    bsdf_microfacet_setup_fresnel_dielectric(0, &b, &c.sd);
    weight = closure_layering_weight(albedo(c, b), weight);
    add(c, b);
    if (!isequal(p.coat_tint, one_spectrum())) {
      float cosNI = dot(c.sd.wi, n);
      float cosNT = sqrtf(1 - sqr(1 / p.coat_ior) * (1 - sqr(cosNI)));
      weight *= mix(one_spectrum(), power(max(p.coat_tint, zero_spectrum()), 1 / cosNT), p.coat);
    }
  }
  c.emission_weight = weight;
}
VN_DEVICE float3 emission_weight(const Parameters &p) {
  Closures c{};
  c.sd.N = p.N; c.sd.Ng = p.Ng; c.sd.wi = p.wi;
  c.sd.flag = SD_USE_BUMP_MAP_CORRECTION;
  float3 weight = one_spectrum();
  setup_layers(p, c, weight);
  return weight;
}
VN_DEVICE void setup(const Parameters &p, Closures &c) {
  c.count = 0; c.total = 0;
  c.subsurface_weight = zero_spectrum();
  c.emission_weight = one_spectrum();
  c.sd = {};
  c.sd.N = p.N; c.sd.Ng = p.Ng; c.sd.wi = p.wi;
  c.sd.flag = SD_USE_BUMP_MAP_CORRECTION;
  const bool multi = p.distribution == MULTI_GGX;
  float ior = p.backfacing ? 1 / p.ior : p.ior;
  float3 n = maybe_ensure_valid_specular_reflection(&c.sd, p.N);
  float3 weight = one_spectrum();
  if (p.model == VELVET) {
    initialize(c.velvet, p.N, p.sheen_tint * p.sheen);
    c.velvet.sigma = p.sheen_roughness;
    bsdf_ashikhmin_velvet_setup(&c.velvet);
    add(c, c.velvet);
    return;
  }
  if (p.model == SHEEN) {
    initialize(c.sheen, p.N, p.sheen_tint * p.sheen);
    c.sheen.roughness = p.sheen_roughness;
    bsdf_sheen_setup(0, &c.sd, &c.sheen);
    add(c, c.sheen);
    return;
  }
  if (p.model == PRINCIPLED) setup_layers(p, c, weight);
  if (p.model == GLOSSY || p.model == GLASS || p.model == REFRACTION) {
    auto &b = c.micro[0];
    setup_micro(b, p, n, p.model == GLASS ? one_spectrum() : p.color,
                p.alpha_x, p.alpha_y, p.model == GLOSSY ? 1 : ior, p.model);
    if (p.model == GLASS) {
      generalized(c, b, c.glass_fresnel, p, make_spectrum(F0_from_ior(p.ior)),
                  p.color, p.color, p.ior);
    } else if (multi && p.model != REFRACTION) {
      bsdf_microfacet_setup_fresnel_constant(0, &b, &c.sd, p.color);
    }
    add(c, b);
    return;
  }
  if (p.model == CONDUCTOR || p.model == METALLIC ||
      (p.model == PRINCIPLED && p.metallic > CLOSURE_WEIGHT_CUTOFF)) {
    auto &b = c.micro[1];
    setup_micro(b, p, n, weight * (p.model == PRINCIPLED ? p.metallic : 1),
                p.alpha_x, p.alpha_y, 1, GLOSSY);
    if (p.model == CONDUCTOR) {
      c.conductor_fresnel.thin_film = film(p, 1);
      c.conductor_fresnel.ior = {max(p.conductor_ior, zero_spectrum()), max(p.conductor_k, zero_spectrum())};
      bsdf_microfacet_setup_fresnel_conductor(0, &b, &c.sd, &c.conductor_fresnel, multi);
    } else {
      c.metal_fresnel.thin_film = film(p, 1);
      c.metal_fresnel.f0 = saturate(p.color);
      bsdf_microfacet_setup_fresnel_f82_tint(0, &b, &c.sd, &c.metal_fresnel,
                                           min(p.specular_tint, one_spectrum()), multi);
    }
    add(c, b);
    if (p.model != PRINCIPLED) return;
    weight *= 1 - p.metallic;
  }
  if (p.model == PRINCIPLED) {
    if (p.transmission > CLOSURE_WEIGHT_CUTOFF) {
      auto &b = c.micro[2];
      setup_micro(b, p, n, weight * p.transmission, p.alpha, p.alpha, ior, GLASS);
      generalized(c, b, c.glass_fresnel, p, F0_from_ior(p.ior) * p.specular_tint,
                  sqrt(saturate(p.color)), one_spectrum(), p.ior);
      add(c, b);
      weight *= 1 - p.transmission;
    }
    float f0 = F0_from_ior(p.ior), eta = p.ior;
    if (p.specular_level != 0.5f) {
      f0 *= 2 * p.specular_level;
      eta = ior_from_F0(f0);
      if (p.ior < 1) eta = 1 / eta;
    }
    if (eta != 1 || p.thin_film_thickness > THINFILM_THICKNESS_CUTOFF) {
      auto &b = c.micro[3];
      setup_micro(b, p, n, weight, p.alpha_x, p.alpha_y, eta, GLOSSY);
      generalized(c, b, c.specular_fresnel, p, f0 * p.specular_tint, zero_spectrum());
      weight = closure_layering_weight(albedo(c, b), weight);
      add(c, b);
    }
    c.subsurface_weight = weight * saturate(p.color) * p.subsurface;
  }
  float3 diffuse_weight = weight * p.color * (1 - p.subsurface);
  if (p.translucent > 0) {
    initialize(c.translucent, p.N, diffuse_weight * p.translucent);
    bsdf_translucent_setup(&c.translucent);
    add(c, c.translucent);
  }
  initialize(c.diffuse, p.N, diffuse_weight * (1 - p.translucent));
  float diffuse_roughness = p.model == DIFFUSE ? p.roughness : p.diffuse_roughness;
  if (diffuse_roughness > CLOSURE_WEIGHT_CUTOFF) {
    c.diffuse.roughness = diffuse_roughness;
    bsdf_oren_nayar_setup(&c.sd, &c.diffuse, p.color);
  } else bsdf_diffuse_setup(reinterpret_cast<DiffuseBsdf *>(&c.diffuse));
  add(c, c.diffuse);
}
VN_INLINE float3 evaluate_one(const Closures &c, const ShaderClosure *sc, float3 wo, float *pdf) {
  *pdf = 0;
  switch (sc->type) {
    case CLOSURE_BSDF_DIFFUSE_ID: return bsdf_diffuse_eval(sc, c.sd.wi, wo, pdf);
    case CLOSURE_BSDF_TRANSLUCENT_ID: return bsdf_translucent_eval(sc, c.sd.wi, wo, pdf);
    case CLOSURE_BSDF_OREN_NAYAR_ID: return bsdf_oren_nayar_eval(sc, c.sd.wi, wo, pdf);
    case CLOSURE_BSDF_SHEEN_ID: return bsdf_sheen_eval(sc, c.sd.wi, wo, pdf);
    case CLOSURE_BSDF_ASHIKHMIN_VELVET_ID: return bsdf_ashikhmin_velvet_eval(sc, c.sd.wi, wo, pdf);
    case CLOSURE_BSDF_MICROFACET_BECKMANN_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID:
      return bsdf_microfacet_beckmann_eval(0, sc, c.sd.wi, wo, pdf);
    default: return bsdf_microfacet_ggx_eval(0, sc, c.sd.wi, wo, pdf);
  }
}
VN_DEVICE float3 evaluate(const Closures &c, float3 wo, float *pdf) {
  float3 value = zero_spectrum(); *pdf = 0;
  for (int i = 0; i < c.count; ++i) {
    const auto *sc = c.entries[i]; float density;
    value += sc->weight * evaluate_one(c, sc, wo, &density);
    *pdf += sc->sample_weight * density;
  }
  if (c.total > 0) *pdf /= c.total;
  return value;
}
VN_DEVICE int sample(const Closures &c, float choose, float3 rand, float3 *wo, float3 *value, float *pdf) {
  *pdf = 0; *value = zero_spectrum();
  *wo = make_float3(0, 0, 1);
  if (c.total <= 0) return LABEL_NONE;
  float remaining = choose * c.total;
  int pick = c.count - 1;
  for (int i = 0; i < c.count - 1; ++i) {
    remaining -= c.entries[i]->sample_weight;
    if (remaining < 0) { pick = i; break; }
  }
  const ShaderClosure *sc = c.entries[pick];
  int label; float2 roughness; float eta;
  switch (sc->type) {
    case CLOSURE_BSDF_DIFFUSE_ID: label = bsdf_diffuse_sample(sc, c.sd.Ng, c.sd.wi, make_float2(rand), value, wo, pdf); break;
    case CLOSURE_BSDF_TRANSLUCENT_ID: label = bsdf_translucent_sample(sc, c.sd.Ng, c.sd.wi, make_float2(rand), value, wo, pdf); break;
    case CLOSURE_BSDF_OREN_NAYAR_ID: label = bsdf_oren_nayar_sample(sc, c.sd.Ng, c.sd.wi, make_float2(rand), value, wo, pdf); break;
    case CLOSURE_BSDF_SHEEN_ID: label = bsdf_sheen_sample(sc, c.sd.Ng, c.sd.wi, make_float2(rand), value, wo, pdf); break;
    case CLOSURE_BSDF_ASHIKHMIN_VELVET_ID: label = bsdf_ashikhmin_velvet_sample(sc, c.sd.Ng, c.sd.wi, make_float2(rand), value, wo, pdf); break;
    case CLOSURE_BSDF_MICROFACET_BECKMANN_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID:
      label = bsdf_microfacet_beckmann_sample(0, sc, c.sd.Ng, c.sd.wi, rand, value, wo, pdf, &roughness, &eta); break;
    default: label = bsdf_microfacet_ggx_sample(0, sc, c.sd.Ng, c.sd.wi, rand, value, wo, pdf, &roughness, &eta); break;
  }
  if (*pdf <= 0 || label == LABEL_NONE) { *pdf = 0; return LABEL_NONE; }
  *value *= sc->weight;
  *pdf *= sc->sample_weight;
  for (int i = 0; i < c.count; ++i) if (i != pick) {
    float density;
    *value += c.entries[i]->weight * evaluate_one(c, c.entries[i], *wo, &density);
    *pdf += c.entries[i]->sample_weight * density;
  }
  *pdf /= c.total;
  return label;
}
} // namespace vibrt_surface
