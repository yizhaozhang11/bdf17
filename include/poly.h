#ifndef POLY_H
#define POLY_H

#include "typed_poly.hpp"

// Legacy include retained for compatibility with include paths.
// Use CoeffPoly<Transform> and EvalPoly<Transform> directly.
template <class Transform>
using PolyCoeff = CoeffPoly<Transform>;

template <class Transform>
using PolyEval = EvalPoly<Transform>;

#endif // POLY_H
