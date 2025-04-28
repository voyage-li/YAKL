
#include "YAKL.h"
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

typedef float real;

typedef Array<real, 1, memHost, styleC> realHost1d;
typedef Array<real, 2, memHost, styleC> realHost2d;

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  Kokkos::DefaultExecutionSpace{}.print_configuration(std::cout);
  yakl::init();
  {
    int constexpr n = 10;

    realHost1d a("a", n);
    realHost1d b("b", n);
    realHost1d c("c", n);

    yakl::c::parallel_for(
        yakl::c::Bounds<1>(n), KOKKOS_LAMBDA(int i) {
          a(i) = 0.f;
          b(i) = 2.f;
          c(i) = 3.f;
        });
    yakl::c::parallel_for(
        yakl::c::Bounds<1>(n), KOKKOS_LAMBDA(int i) { a(i) = b(i) + c(i); });

    for (int i = 0; i < 10; i++) {
      std::cout << "i: " << a(i) << "\n";
    }

    realHost2d A("A", n, n);
    realHost2d B("B", n, n);
    realHost2d C("C", n, n);

    yakl::c::parallel_for(
        yakl::c::Bounds<2>(n, n), KOKKOS_LAMBDA(int i, int j) {
          A(i, j) = 0.f;
          B(i, j) = 2.f;
          C(i, j) = 3.f;
        });
    yakl::c::parallel_for(
        yakl::c::Bounds<2>(n, n),
        KOKKOS_LAMBDA(int i, int j) { A(i, j) = B(i, j) + C(i, j); });

    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 10; j++) {
        std::cout << A(i, j) << " ";
      }
      std::cout << std::endl;
    }
  }

  yakl::finalize();

  return 0;
}
