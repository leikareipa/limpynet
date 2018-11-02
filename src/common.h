/*
 * 2018 Tarpeeksi Hyvae Soft
 *
 */

#ifndef COMMON_H
#define COMMON_H

#include <cassert>
#include <cstdio>
#include "../src/types.h"

#define k_assert(condition, error_string)  assert(condition && error_string);

#define INFO(args)  (printf("[info ] {%s:%i} ", __FILE__, __LINE__), printf args, printf("\n"), fflush(stdout))
#define DEBUG(args) (printf("[debug] {%s:%i} ", __FILE__, __LINE__), printf args, printf("\n"), fflush(stdout))
#define NBENE(args) (fprintf(stderr, "[ERROR] {%s:%i} ", __FILE__, __LINE__), printf args, printf("\n"), fflush(stdout));

#endif
