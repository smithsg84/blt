// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other BLT Project Developers. See the top-level LICENSE file for details
//
// SPDX-License-Identifier: (BSD-3-Clause)

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Note: This is a CUDA example from NVIDIA:
// Obtained from here:
// https://devblogs.nvidia.com/parallelforall/how-query-device-properties-and-handle-errors-cuda-cc/
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
//
// file: blt_cuda_runtime_smoke.cpp
//
//-----------------------------------------------------------------------------

#include <cstdio>
#include <cuda_runtime.h>

static bool checkCuda(cudaError_t err, const char* what)
{
  if (err != cudaSuccess) {
    std::printf("%s failed: %s\n", what, cudaGetErrorString(err));
    return false;
  }
  return true;
}

int main()
{
  int nDevices = 0;
  if (!checkCuda(cudaGetDeviceCount(&nDevices), "cudaGetDeviceCount")) {
    return 1;
  }

  for (int i = 0; i < nDevices; ++i)
  {
    cudaDeviceProp prop{};
    if (!checkCuda(cudaGetDeviceProperties(&prop, i), "cudaGetDeviceProperties")) {
      continue;
    }

    int memoryClockRateKHz = 0;
    int memoryBusWidthBits = 0;

    if (!checkCuda(cudaDeviceGetAttribute(&memoryClockRateKHz,
                                          cudaDevAttrMemoryClockRate,
                                          i),
                   "cudaDeviceGetAttribute(cudaDevAttrMemoryClockRate)")) {
      continue;
    }

    if (!checkCuda(cudaDeviceGetAttribute(&memoryBusWidthBits,
                                          cudaDevAttrGlobalMemoryBusWidth,
                                          i),
                   "cudaDeviceGetAttribute(cudaDevAttrGlobalMemoryBusWidth)")) {
      continue;
    }

    double peakBandwidthGBs =
      2.0 * static_cast<double>(memoryClockRateKHz) *
      (static_cast<double>(memoryBusWidthBits) / 8.0) / 1.0e6;

    std::printf("Device Number: %d\n", i);
    std::printf("  Device name: %s\n", prop.name);
    std::printf("  Memory Clock Rate (KHz): %d\n", memoryClockRateKHz);
    std::printf("  Memory Bus Width (bits): %d\n", memoryBusWidthBits);
    std::printf("  Peak Memory Bandwidth (GB/s): %f\n\n", peakBandwidthGBs);
  }

  return 0;
}

