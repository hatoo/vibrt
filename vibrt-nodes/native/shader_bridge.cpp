#include "types.h"

#include "core.h"
#include "cycles_nodes.h"
#include "shader.h"
extern "C" float vibrt_shader_math(unsigned int op, float a, float b, float c) {
  return vibrt_cycles::svm_math((vibrt_cycles::NodeMathType)op, a, b, c);
}
extern "C" void vibrt_shader_eval(const ShaderInstruction *nodes, unsigned int count,
                                  const float *context, float *output, unsigned int requested) {
  vibrt_shader::evaluate(nodes, count, reinterpret_cast<const vibrt_cycles::float4 *>(context),
                         reinterpret_cast<vibrt_cycles::float4 *>(output), requested);
}
