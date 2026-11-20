#include "types.h"

#include "core.h"
#include "cycles_nodes.h"
#include "shader.h"
extern "C" float vibrt_shader_math(unsigned int op, float a, float b, float c) {
  return vibrt_cycles::svm_math((vibrt_cycles::NodeMathType)op, a, b, c);
}
extern "C" float vibrt_phase_pdf(unsigned int type, const float *parameters,
                                 float g, float cosine) {
  return vibrt_shader::volume_phase_pdf(type, parameters, g, cosine);
}
extern "C" void vibrt_phase_sample(unsigned int type, const float *parameters,
                                   float g, float u, float v, float *result) {
  float pdf;
  auto d = vibrt_shader::volume_phase_sample(type, parameters, g, u, v, &pdf);
  result[0] = d.x;
  result[1] = d.y;
  result[2] = d.z;
  result[3] = pdf;
}
extern "C" void vibrt_shader_eval(const ShaderInstruction *nodes, unsigned int count,
                                  const float *context, float *output, unsigned int requested) {
  vibrt_shader::evaluate(nodes, count, reinterpret_cast<const vibrt_cycles::float4 *>(context),
                         reinterpret_cast<vibrt_cycles::float4 *>(output), requested);
}
