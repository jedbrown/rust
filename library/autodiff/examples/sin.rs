use autodiff::autodiff;

// This syntax defines a new function called `cos_inplace` by differentiating the code below.
#[autodiff(cos_inplace, Reverse, Const)]
fn sin_inplace(#[dup] x: &f32, #[dup] y: &mut f32) {
    *y = x.sin();
}

// This syntax (with only a function declaration) differentiates `sin_inplace`
// and makes the result callable as `cos_inplace_fwd`.
//
// Const refers to the (non-existent in this case) return value. In forward
// mode, we duplicate the inputs x and read the result from dy, without a need
// for y (which is undefined because we use `dup_noneed`).
#[autodiff(sin_inplace, Forward, Const)]
fn cos_inplace_fwd1(#[dup] x: &f32, dx: &f32, #[dup_noneed] y: &mut f32, dy: &mut f32);

// This has identical semantics to above, but specifies dup semantics up-front
// instead of using inline attribute macros.
#[autodiff(sin_inplace, Forward, Const, Duplicated, DuplicatedNoNeed)]
fn cos_inplace_fwd2(x: &f32, dx: &f32, y: &mut f32, dy: &mut f32);

fn main() {
    // Here we can use ==, even though we work on f32.
    // Enzyme will recognize the sin function and replace it with llvm's cos function (see below).
    // Calling f32::cos directly will also result in calling llvm's cos function.
    let a = std::f32::consts::PI / 6.0;
    let mut da = 0.0;
    let mut y = 0.0;
    cos_inplace(&a, &mut da, &mut y, &mut 1.0);

    dbg!(&a, &da, &y);
    assert!(da - f32::cos(a) == 0.0);

    // To use forward mode, we specify da and recover the result from dy. The
    // result value of y is undefined due to DuplicatedNoNeed.
    let mut dy = 0.0;
    da = 1.0;
    y = 0.0;
    cos_inplace_fwd1(&a, &da, &mut y, &mut dy);
    dbg!(&a, &da, &dy);
    assert!(dy - f32::cos(a) == 0.0);

    // Same as above using the second syntax
    cos_inplace_fwd2(&a, &da, &mut y, &mut dy);
    dbg!(&a, &da, &dy);
    assert!(dy - f32::cos(a) == 0.0);
}

// Just for curious readers, this is the (inner) function that Enzyme does generate:
// define internal { float } @diffe_ZN3sin3sin17h18f17f71fe94e58fE(float %0, float %1) unnamed_addr #35 {
//   %3 = call fast float @llvm.cos.f32(float %0)
//   %4 = fmul fast float %1, %3
//   %5 = insertvalue { float } undef, float %4, 0
//   ret { float } %5
// }

#[cfg(test)]
mod tests {
    #[test]
    fn main() {
        super::main()
    }
}
