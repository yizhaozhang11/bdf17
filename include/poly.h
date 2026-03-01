#ifndef POLY_H
#define POLY_H

#include "typed_poly.hpp"

// Compatibility-only include retained for existing call sites.
// New code should use CoeffPoly<Transform> and EvalPoly<Transform> directly.
template <class Transform>
using PolyCoeff = CoeffPoly<Transform>;

template <class Transform>
using PolyEval = EvalPoly<Transform>;

#endif // POLY_H
