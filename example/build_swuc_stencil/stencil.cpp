
#include "YAKL.h"
#include "YAKL_timers.h"
#include <algorithm>
#include <iostream>
#include <random>

using yakl::Array;
using yakl::COLON;
using yakl::memDevice;
using yakl::memHost;
using yakl::styleC;
using yakl::c::Bounds;
using yakl::c::parallel_for;
using yakl::c::SimpleBounds;

typedef double real;

typedef Array<real, 1, memHost, styleC> realHost1d;
typedef Array<real, 2, memHost, styleC> realHost2d;
typedef Array<real, 3, memHost, styleC> realHost3d;

#define randq0 0.42902885331071894
#define randq1 0.3835149150254443
#define randq2 0.23069394385027253
#define randq3 0.8490940280251191
#define randq4 0.13880139717450857
#define randq5 0.7535669344593772
#define randq6 0.9871726780384124
#define randq7 0.6517359346213617
#define randq8 0.007027172843024565
#define randq9 0.18381318805149127
#define randq10 0.136064001401214
#define randq11 0.8714124615900947
#define randq12 0.8714348784956923
#define randq13 0.3958204533718198
#define randq14 0.2850986365562863
#define randq15 0.8170168349259794
#define randq16 0.847846769248984
#define randq17 0.6256118416818361
#define randq18 0.23735018356951787
#define randq19 0.5933040715916058
#define randq20 0.06763653209489273
#define randq21 0.4832035201020275
#define randq22 0.041225810690153075
#define randq23 0.030359155660743875
#define randq24 0.24026628330922173
#define randq25 0.8768136077897237
#define randq26 0.13218452098797873

template <typename T>
bool cr(const T& a, const T& b) {
  return !(fabs(a - b) < 1e-5);
}

void stencil2d5p(int N) {
  printf("stencil2d5p N = %d\n", N);
  realHost2d a("a", N + 2, N + 2);
  realHost2d b1("b1", N + 2, N + 2);
  realHost2d b2("b2", N + 2, N + 2);

  for (int i = 0; i < N + 2; i++) {
    for (int j = 0; j < N + 2; j++) {
      a(i, j) = (double)rand() / RAND_MAX;
    }
  }
  for (int i = 0; i < 20; i++) {
    yakl::timer_start("stencil2d5p serial");
    for (int i = 1; i <= N; i++) {
      for (int j = 1; j <= N; j++) {
        b1(i, j) = randq0 * a(i, j) + randq1 * a(i - 1, j) + randq2 * a(i + 1, j) + randq3 * a(i, j - 1) + randq4 * a(i, j + 1);
      }
    }
    yakl::timer_stop("stencil2d5p serial");
    yakl::timer_start("stencil2d5p parallel");
    yakl::c::parallel_for(
        yakl::c::Bounds<2>({1, N}, {1, N}), KOKKOS_LAMBDA(int i, int j) {
          b2(i, j) = randq0 * a(i, j) + randq1 * a(i - 1, j) + randq2 * a(i + 1, j) + randq3 * a(i, j - 1) + randq4 * a(i, j + 1);
        });
    Kokkos::fence();
    yakl::timer_stop("stencil2d5p parallel");
  }
  // check result
  for (int i = 1; i <= N; i++) {
    for (int j = 1; j <= N; j++) {
      if (cr(b1(i, j), b2(i, j))) {
        printf("Wrong result: %f != %f, index %d %d\n", b1(i, j), b2(i, j), i, j);
        break;
      }
    }
  }
}

void stencil2d9p(int N) {
  printf("stencil2d9p N = %d\n", N);
  realHost2d a("a", N + 2, N + 2);
  realHost2d b1("b1", N + 2, N + 2);
  realHost2d b2("b2", N + 2, N + 2);

  for (int i = 0; i < N + 2; i++) {
    for (int j = 0; j < N + 2; j++) {
      a(i, j) = (double)rand() / RAND_MAX;
    }
  }
  for (int i = 0; i < 20; i++) {
    yakl::timer_start("stencil2d9p serial");
    for (int i = 1; i <= N; i++) {
      for (int j = 1; j <= N; j++) {
        b1(i, j) = randq0 * a(i, j) + randq1 * a(i - 1, j) + randq2 * a(i + 1, j) + randq3 * a(i, j - 1) + randq4 * a(i, j + 1) +
                   randq5 * a(i - 1, j - 1) + randq6 * a(i - 1, j + 1) + randq7 * a(i + 1, j - 1) + randq8 * a(i + 1, j + 1);
      }
    }
    yakl::timer_stop("stencil2d9p serial");
    yakl::timer_start("stencil2d9p parallel");
    yakl::c::parallel_for(
        yakl::c::Bounds<2>({1, N}, {1, N}), KOKKOS_LAMBDA(int i, int j) {
          b2(i, j) = randq0 * a(i, j) + randq1 * a(i - 1, j) + randq2 * a(i + 1, j) + randq3 * a(i, j - 1) +
                     randq4 * a(i, j + 1) + randq5 * a(i - 1, j - 1) + randq6 * a(i - 1, j + 1) + randq7 * a(i + 1, j - 1) + randq8 * a(i + 1, j + 1);
        });
    Kokkos::fence();
    yakl::timer_stop("stencil2d9p parallel");
  }
  // check result
  for (int i = 1; i <= N; i++) {
    for (int j = 1; j <= N; j++) {
      if (cr(b1(i, j), b2(i, j))) {
        printf("Wrong result: %f != %f, index %d %d\n", b1(i, j), b2(i, j), i, j);
        break;
      }
    }
  }
}

void stencil3d7p(int N) {
  printf("stencil3d7p N = %d\n", N);
  realHost3d a("a", N + 2, N + 2, N + 2);
  realHost3d b1("b1", N + 2, N + 2, N + 2);
  realHost3d b2("b2", N + 2, N + 2, N + 2);

  for (int i = 0; i < N + 2; i++) {
    for (int j = 0; j < N + 2; j++) {
      for (int k = 0; k < N + 2; k++) {
        a(i, j, k) = (double)rand() / RAND_MAX;
      }
    }
  }
  for (int i = 0; i < 20; i++) {
    yakl::timer_start("stencil3d7p serial");
    for (int i = 1; i <= N; i++) {
      for (int j = 1; j <= N; j++) {
        for (int k = 1; k <= N; k++) {
          b1(i, j, k) = randq0 * a(i, j, k) + randq1 * a(i - 1, j, k) + randq2 * a(i + 1, j, k) +
                        randq3 * a(i, j - 1, k) + randq4 * a(i, j + 1, k) +
                        randq5 * a(i, j, k - 1) + randq6 * a(i, j, k + 1);
        }
      }
    }
    yakl::timer_stop("stencil3d7p serial");
    yakl::timer_start("stencil3d7p parallel");
    yakl::c::parallel_for(
        yakl::c::Bounds<3>({1, N}, {1, N}, {1, N}), KOKKOS_LAMBDA(int i, int j, int k) {
          b2(i, j, k) = randq0 * a(i, j, k) + randq1 * a(i - 1, j, k) + randq2 * a(i + 1, j, k) +
                        randq3 * a(i, j - 1, k) + randq4 * a(i, j + 1, k) +
                        randq5 * a(i, j, k - 1) + randq6 * a(i, j, k + 1);
        });
    Kokkos::fence();
    yakl::timer_stop("stencil3d7p parallel");
  }
  // check result
  for (int i = 1; i <= N; i++) {
    for (int j = 1; j <= N; j++) {
      for (int k = 1; k <= N; k++) {
        if (cr(b1(i, j, k), b2(i, j, k))) {
          printf("Wrong result: %f != %f, index %d %d %d\n", b1(i, j, k), b2(i, j, k), i, j, k);
          break;
        }
      }
    }
  }
}

void stencil3d27p(int N) {
  printf("stencil3d27p N = %d\n", N);
  realHost3d a("a", N + 2, N + 2, N + 2);
  realHost3d b1("b1", N + 2, N + 2, N + 2);
  realHost3d b2("b2", N + 2, N + 2, N + 2);

  for (int i = 0; i < N + 2; i++) {
    for (int j = 0; j < N + 2; j++) {
      for (int k = 0; k < N + 2; k++) {
        a(i, j, k) = (double)rand() / RAND_MAX;
      }
    }
  }
  for (int i = 0; i < 20; i++) {
    yakl::timer_start("stencil3d27p serial");
    for (int i = 1; i <= N; i++) {
      for (int j = 1; j <= N; j++) {
        for (int k = 1; k <= N; k++) {
          b1(i, j, k) = randq0 * a(i, j, k) + randq1 * a(i - 1, j, k) + randq2 * a(i + 1, j, k) +
                        randq3 * a(i, j - 1, k) + randq4 * a(i, j + 1, k) +
                        randq5 * a(i, j, k - 1) + randq6 * a(i, j, k + 1) +
                        randq7 * a(i - 1, j - 1, k) + randq8 * a(i - 1, j + 1, k) +
                        randq9 * a(i + 1, j - 1, k) + randq10 * a(i + 1, j + 1, k) +
                        randq11 * a(i - 1, j, k - 1) + randq12 * a(i - 1, j, k + 1) +
                        randq13 * a(i + 1, j, k - 1) + randq14 * a(i + 1, j, k + 1) +
                        randq15 * a(i, j - 1, k - 1) + randq16 * a(i, j - 1, k + 1) +
                        randq17 * a(i, j + 1, k - 1) + randq18 * a(i, j + 1, k + 1) +
                        randq19 * a(i - 1, j - 1, k - 1) + randq20 * a(i - 1, j - 1, k + 1) +
                        randq21 * a(i - 1, j + 1, k - 1) + randq22 * a(i - 1, j + 1, k + 1) +
                        randq23 * a(i + 1, j - 1, k - 1) + randq24 * a(i + 1, j - 1, k + 1) +
                        randq25 * a(i + 1, j + 1, k - 1) + randq26 * a(i + 1, j + 1, k + 1);
        }
      }
    }
    yakl::timer_stop("stencil3d27p serial");
    yakl::timer_start("stencil3d27p parallel");
    yakl::c::parallel_for(
        yakl::c::Bounds<3>({1, N}, {1, N}, {1, N}), KOKKOS_LAMBDA(int i, int j, int k) {
          b2(i, j, k) = randq0 * a(i, j, k) + randq1 * a(i - 1, j, k) + randq2 * a(i + 1, j, k) +
                        randq3 * a(i, j - 1, k) + randq4 * a(i, j + 1, k) +
                        randq5 * a(i, j, k - 1) + randq6 * a(i, j, k + 1) +
                        randq7 * a(i - 1, j - 1, k) + randq8 * a(i - 1, j + 1, k) +
                        randq9 * a(i + 1, j - 1, k) + randq10 * a(i + 1, j + 1, k) +
                        randq11 * a(i - 1, j, k - 1) + randq12 * a(i - 1, j, k + 1) +
                        randq13 * a(i + 1, j, k - 1) + randq14 * a(i + 1, j, k + 1) +
                        randq15 * a(i, j - 1, k - 1) + randq16 * a(i, j - 1, k + 1) +
                        randq17 * a(i, j + 1, k - 1) + randq18 * a(i, j + 1, k + 1) +
                        randq19 * a(i - 1, j - 1, k - 1) + randq20 * a(i - 1, j - 1, k + 1) +
                        randq21 * a(i - 1, j + 1, k - 1) + randq22 * a(i - 1, j + 1, k + 1) +
                        randq23 * a(i + 1, j - 1, k - 1) + randq24 * a(i + 1, j - 1, k + 1) +
                        randq25 * a(i + 1, j + 1, k - 1) + randq26 * a(i + 1, j + 1, k + 1);
        });
    Kokkos::fence();
    yakl::timer_stop("stencil3d27p parallel");
  }
  // check result
  for (int i = 1; i <= N; i++) {
    for (int j = 1; j <= N; j++) {
      for (int k = 1; k <= N; k++) {
        if (cr(b1(i, j, k), b2(i, j, k))) {
          printf("Wrong result: %f != %f, index %d %d %d\n", b1(i, j, k), b2(i, j, k), i, j, k);
          break;
        }
      }
    }
  }
}

void heat2d(int N) {
  printf("heat2d N = %d\n", N);
  realHost2d a("a", N + 2, N + 2);
  realHost2d b1("b1", N + 2, N + 2);
  realHost2d b2("b2", N + 2, N + 2);

  for (int i = 0; i < N + 2; i++) {
    for (int j = 0; j < N + 2; j++) {
      a(i, j) = (double)rand() / RAND_MAX;
    }
  }
  for (int i = 0; i < 20; i++) {
    yakl::timer_start("heat-2d serial");
    for (int i = 1; i <= N; i++) {
      for (int j = 1; j <= N; j++) {
        b1(i, j) = 0.125 * (a(i + 1, j) - 2.0 * a(i, j) + a(i - 1, j)) +
                   0.125 * (a(i, j + 1) - 2.0 * a(i, j) + a(i, j - 1)) +
                   a(i, j);
      }
    }
    yakl::timer_stop("heat-2d serial");
    yakl::timer_start("heat-2d parallel");
    yakl::c::parallel_for(
        yakl::c::Bounds<2>({1, N}, {1, N}), KOKKOS_LAMBDA(int i, int j) {
          b2(i, j) = 0.125 * (a(i + 1, j) - 2.0 * a(i, j) + a(i - 1, j)) +
                     0.125 * (a(i, j + 1) - 2.0 * a(i, j) + a(i, j - 1)) +
                     a(i, j);
        });
    Kokkos::fence();
    yakl::timer_stop("heat-2d parallel");
  }
  // check result
  for (int i = 1; i <= N; i++) {
    for (int j = 1; j <= N; j++) {
      if (cr(b1(i, j), b2(i, j))) {
        printf("Wrong result: %f != %f, index %d %d\n", b1(i, j), b2(i, j), i, j);
        break;
      }
    }
  }
}

void jacobi2d(int N) {
  printf("jacobi2d N = %d\n", N);
  realHost2d a("a", N + 2, N + 2);
  realHost2d b1("b1", N + 2, N + 2);
  realHost2d b2("b2", N + 2, N + 2);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      a(i, j) = (static_cast<double>(j) / N);
    }
  }
  for (int i = 0; i < 20; i++) {
    yakl::timer_start("jacobi-2d serial");
    for (int i = 2; i < N - 1; i++) {
      for (int j = 2; j < N - 1; j++) {
        b1(i, j) = 0.2 * (a(i, j) + a(i, j - 1) + a(i, j + 1) + a(i + 1, j) + a(i - 1, j));
      }
    }
    yakl::timer_stop("jacobi-2d serial");
    yakl::timer_start("jacobi-2d parallel");
    yakl::c::parallel_for(
        yakl::c::Bounds<2>({2, N - 1}, {2, N - 1}), KOKKOS_LAMBDA(int i, int j) {
          b2(i, j) = 0.2 * (a(i, j) + a(i, j - 1) + a(i, j + 1) + a(i + 1, j) + a(i - 1, j));
        });
    Kokkos::fence();
    yakl::timer_stop("jacobi-2d parallel");
  }
  // check result
  for (int i = 2; i < N - 1; i++) {
    for (int j = 2; j < N - 1; j++) {
      if (cr(b1(i, j), b2(i, j))) {
        printf("Wrong result: %f != %f, index %d %d\n", b1(i, j), b2(i, j), i, j);
        break;
      }
    }
  }
}

void fdtd2d(int N, int NY) {
  printf("fdtd2d N = %d NY = %d\n", N, NY);
  realHost2d ex1("ex1", N + 1, NY + 1), ey1("ey1", N + 1, NY + 1), hz1("hz1", N + 1, NY + 1);
  realHost2d ex2("ex2", N + 1, NY + 1), ey2("ey2", N + 1, NY + 1), hz2("hz2", N + 1, NY + 1);

  for (int i = 0; i < N + 1; i++) {
    for (int j = 0; j < NY + 1; j++) {
      if (i < N) {
        ey1(i, j) = static_cast<double>(j) / N;
        ey2(i, j) = static_cast<double>(j) / N;
      } else {
        ey1(i, j) = 0.0;
        ey2(i, j) = 0.0;
      }

      ex1(i, j) = 0.0;
      ex2(i, j) = 0.0;

      hz1(i, j) = 0.0;
      hz2(i, j) = 0.0;
    }
  }
  for (int t = 0; t < 128; t++) {
    yakl::timer_start("fdtd2d serial");
    for (int j = 0; j < NY; j++) {
      ey1(0, j) = t;
    }
    for (int i = 1; i < N; i++) {
      for (int j = 0; j < NY; j++) {
        ey1(i, j) = ey1(i, j) - 0.5 * (hz1(i, j) - hz1(i - 1, j));
      }
    }
    for (int i = 0; i < N; i++) {
      for (int j = 1; j < NY; j++) {
        ex1(i, j) = ex1(i, j) - 0.5 * (hz1(i, j) - hz1(i, j - 1));
      }
    }
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < NY; j++) {
        hz1(i, j) = hz1(i, j) - 0.7 * (ex1(i, j + 1) - ex1(i, j) + ey1(i + 1, j) - ey1(i, j));
      }
    }
    yakl::timer_stop("fdtd2d serial");
    yakl::timer_start("fdtd2d parallel");
    yakl::c::parallel_for(NY, KOKKOS_LAMBDA(int j) { ey2(0, j) = t; });
    Kokkos::fence();

    yakl::c::parallel_for(
        yakl::c::Bounds<2>({0, N - 1}, {0, NY - 1}), KOKKOS_LAMBDA(int i, int j) {
          if (i >= 1) ey2(i, j) = ey2(i, j) - 0.5 * (hz2(i, j) - hz2(i - 1, j));
          if (j >= 1) ex2(i, j) = ex2(i, j) - 0.5 * (hz2(i, j) - hz2(i, j - 1));
        });
    Kokkos::fence();

    yakl::c::parallel_for(
        yakl::c::Bounds<2>({0, N - 1}, {0, NY - 1}), KOKKOS_LAMBDA(int i, int j) {
          hz2(i, j) = hz2(i, j) - 0.7 * (ex2(i, j + 1) - ex2(i, j) + ey2(i + 1, j) - ey2(i, j));
        });
    Kokkos::fence();

    yakl::timer_stop("fdtd2d parallel");
  }
  // check result
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < NY; j++) {
      if (cr(hz1(i, j), hz2(i, j))) {
        printf("Wrong result: %f != %f, index %4d %4d\n", hz1(i, j), hz2(i, j), i, j);
        break;
      }
    }
  }
}

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  Kokkos::DefaultExecutionSpace{}.print_configuration(std::cout);
  yakl::init();
  {
    if (argc != 3) {
      printf("Usage: %s <N> <N3D>\n", argv[0]);
      exit(1);
    }
    const int N   = atoi(argv[1]);
    const int N3D = atoi(argv[2]);

    printf("N = %d\t", N);
    printf("N3D = %d\n", N3D);
    stencil2d5p(N);
    stencil2d9p(N);
    stencil3d7p(N3D);
    stencil3d27p(N3D);
    heat2d(N);
    jacobi2d(N);
    fdtd2d(N, N);
    // yakl::c::parallel_for(
    //     yakl::c::Bounds<2>({0, N}, {0, N}), KOKKOS_LAMBDA(int i, int j) {
    //       if (i == N || j == N) {
    //         printf("yakl from thread %d %d\n", i, j);
    //       }
    //     });
    // Kokkos::fence();
    // Kokkos::parallel_for(
    //     Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, N}), KOKKOS_LAMBDA(int i, int j) {
    //       if (i == N || j == N) {
    //         printf("kokkos from thread %d %d\n", i, j);
    //       }
    //     });
  }

  yakl::finalize();

  return 0;
}
