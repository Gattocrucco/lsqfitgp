#include "fasthash.h"
#include <stdio.h>

int main() {
    uint8_t input0[] = {};
    uint8_t input1[] = {234};
    uint8_t input2[] = {194, 116};
    uint8_t input3[] = {160, 237, 166};
    uint8_t input4[] = {72, 56, 46, 113};
    uint8_t input5[] = {152, 22, 163, 7, 234};
    uint8_t input6[] = {100, 11, 190, 249, 103, 74};
    uint8_t input7[] = {119, 52, 46, 248, 188, 178, 216};
    uint8_t input8[] = {81, 10, 197, 4, 19, 41, 69, 164};
    uint8_t input9[] = {53, 246, 128, 162, 79, 228, 71, 137, 255};
    uint8_t input10[] = {145, 141, 43, 100, 125, 107, 12, 4, 147, 229};
    uint8_t input11[] = {117, 92, 35, 144, 76, 140, 59, 36, 42, 13, 94};
    uint8_t input12[] = {91, 207, 0, 152, 226, 159, 190, 164, 136, 176, 194, 59};
    uint8_t input13[] = {126, 94, 132, 168, 44, 150, 242, 165, 199, 149, 248, 82, 141};
    uint8_t input14[] = {26, 101, 134, 203, 216, 141, 100, 242, 248, 225, 83, 131, 27, 100};
    uint8_t input15[] = {153, 2, 211, 91, 131, 54, 101, 233, 213, 71, 216, 126, 60, 48, 157};
    uint8_t input16[] = {114, 165, 8, 26, 213, 17, 112, 170, 104, 161, 164, 95, 53, 17, 149, 170};
    uint8_t input17[] = {40, 198, 242, 87, 28, 55, 234, 142, 22, 200, 236, 65, 198, 91, 197, 233, 46};
    uint8_t input18[] = {208, 21, 5, 101, 61, 240, 41, 134, 164, 25, 109, 253, 108, 140, 229, 255, 39, 199};
    uint8_t input19[] = {240, 22, 57, 231, 226, 172, 97, 114, 34, 20, 14, 47, 118, 129, 193, 93, 43, 209, 75};
    uint8_t *inputs[] = {input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12, input13, input14, input15, input16, input17, input18, input19};
    uint32_t seed32 = 2428169863U;
    uint64_t seed64 = 6361217807637034346ULL;
    int i;
    printf("    hashes64 = jnp.array([\n");
    for (i = 0; i < sizeof(inputs) / sizeof(inputs[0]); ++i) {
        printf("        %llu,\n", fasthash64((void *)inputs[i], i, seed64));
    }
    printf("    ], dtype=jnp.uint64)\n");
    printf("    hashes32 = jnp.array([\n");
    for (i = 0; i < sizeof(inputs) / sizeof(inputs[0]); ++i) {
        printf("        %u,\n", fasthash32((void *)inputs[i], i, seed32));
    }
    printf("    ], dtype=jnp.uint32)\n");
}
