/*
 * 2016-2018 Tarpeeksi Hyvae Soft
 *
 * A basic C-based file access wrapper.
 *
 */

#ifndef FILE_H
#define FILE_H

#include "../../src/common.h"

// For converting between big and little endian.
template <typename T>
T byteswap(const T val)
{
    T swapped = val;

    switch (sizeof(T))
    {
        case 1: swapped = __builtin_bswap32(swapped << 24); break;
        case 2: swapped = __builtin_bswap32(swapped << 16); break;
        case 4: swapped = __builtin_bswap32(swapped); break;
        case 8: swapped = __builtin_bswap64(swapped); break;
        default: k_assert(0, "Failed to byteswap the given value - unsupported bit count."); break;
    }

    return swapped;
}

typedef u16 file_handle_t;

file_handle_t kfile_open_file(const char *const filename, const char *const mode);

void kfile_close_file(const file_handle_t handle);

void kfile_seek(const u32 pos, const file_handle_t handle);

void kfile_jump(const i32 posDelta, const file_handle_t handle);

u32 kfile_file_size(const file_handle_t handle);

FILE* kfile_exposed_file_handle(file_handle_t handle);

template <typename T>
T kfile_read_value(const file_handle_t handle, bool littleEndian = true)
{
    T v = 0;

    const size_t r = fread((u8*)&v, sizeof(T), 1, kfile_exposed_file_handle(handle));
    k_assert(r == 1, "Failed to read a variable from the file.");

    return littleEndian? v : byteswap<T>(v);
}

template <typename T>
T kfile_peek_value(const file_handle_t handle, bool littleEndian = true)
{
    const T v = kfile_read_value<T>(handle);
    kfile_jump(-int(sizeof(T)), handle);

    return littleEndian? v : byteswap<T>(v);
}

template <typename T>
void kfile_write_value(const T v, const file_handle_t handle, bool littleEndian = true)
{
    k_assert(littleEndian, "Big-endian writing is not supported.");

    const size_t r = fwrite((u8*)&v, sizeof(T), 1, kfile_exposed_file_handle(handle));
    k_assert(r == 1, "Failed to write the given data to file.");

    return;
}

void kfile_read_byte_array(u8 *dst, const size_t numBytes, const file_handle_t handle);

void kfile_create_directory(const char *const name, const bool warnIfExists);

void kfile_seek(const u32 pos, const file_handle_t handleId);

bool kfile_getline(const file_handle_t handle, char *const dst, const size_t maxLen);

void kfile_append_contents(const file_handle_t fhSrc, const file_handle_t fhDst);

long kfile_position(const file_handle_t handle);

void kfile_rewind_file(const file_handle_t handle);

void kfile_flush_file(const file_handle_t handle);

void kfile_fill(const unsigned char byte, const unsigned long len, const file_handle_t handle);

void kfile_write_byte_array(const unsigned char *const src, const unsigned long len, const file_handle_t handle);

void kfile_write_string(const char *const str, const file_handle_t handle);

#endif
