# Library for 1D Gaussian messages and distribution
#
# 2023 by Ralf Herbrich
# Hasso-Plattner Institute

"""
Data structure that captures the state of an normalized 1D Gaussian. 
In this represenation, we are storing the precision times mean (tau) and the 
precision (rho). This representations allows for numerically stable products of 
1D-Gaussians.

Should contain two variables:
tau and rho, both Floats, to store the natural parameters of the gaussian.
"""
struct Gaussian1D
    tau::Float64
    rho::Float64

    # default constructor
    Gaussian1D(tau, rho) = rho >= 0 ? new(tau, rho) : error("precision should not be less than 0")
end
# Initializes a standard Gaussian 
Gaussian1D() = Gaussian1D(0, 1)

"""
    Gaussian1DFromMeanVariance(μ,σ2)

Initializes a Gaussian from mean and variance.
"""
Gaussian1DFromMeanVariance(μ, σ2) = Gaussian1D(μ / σ2, μ / σ2)

"""
    mean(g)

Returns the mean of the 1D-Gaussian
```julia-repl
julia> mean(Gaussian1D(1,2))
0.5

julia> mean(Gaussian1DFromMeanVariance(1,2))
1.0
```
"""
mean(g::Gaussian1D) = g.tau / g.rho

"""
    variance(g)

Returns the variance of the 1D-Gaussian 
```julia-repl
julia> variance(Gaussian1D(1,2))
0.5

julia> variance(Gaussian1DFromMeanVariance(1,2))
2.0
```
"""
variance(g::Gaussian1D) = 1 / g.rho


"""
    absdiff(g1,g2)

Computes the absolute difference of `g1` and `g2` in terms of the maximum of |tau_1-tau_2| and sqrt(|rho_1-rho_2|).
# Examples
```julia-repl
julia> absdiff(Gaussian1D(0,1),Gaussian1D(0,2))
1.0

julia> absdiff(Gaussian1D(0,1),Gaussian1D(0,3))
1.4142135623730951
```
"""
absdiff(g1::Gaussian1D, g2::Gaussian1D) = max(abs(g1.tau - g2.tau), sqrt(abs(g1.rho - g2.rho)))

"""
    *(g1,g2)

Multiplies two 1D Gaussians together and re-normalizes them
# Examples
```julia-repl
julia> Gaussian1D() * Gaussian1D()
μ = 0.0, σ = 0.7071067811865476
```
"""
function Base.:*(g1::Gaussian1D, g2::Gaussian1D)
    return Gaussian1D(g1.tau + g2.tau, g1.rho + g2.rho)
end

"""
    /(g1,g2)

Divides two 1D Gaussians from each other
# Examples
```julia-repl
julia> Gaussian1D(0,1) / Gaussian1D(0,0.5)
μ = 0.0, σ = 1.4142135623730951
```
"""
function Base.:/(g1::Gaussian1D, g2::Gaussian1D)
    return Gaussian1D(g1.tau - g2.tau, g1.rho - g2.rho)
end

"""
    logNormProduct(g1,g2)

Computes the log-normalization constant of a multiplication of `g1` and `g2` (the end of the equation ;))

It should be 0 if both rho variables are 0.
# Examples
```julia-repl
julia> logNormProduct(Gaussian1D() * Gaussian1D())
c = 0.28209479177387814
```
"""
function logNormProduct(g1::Gaussian1D, g2::Gaussian1D)
    if (g1.rho == g2.rho == 0.0)
        return 0.0;
    end
    x = mean(g1)
    m = mean(g2)
    v = variance(g1) + variance(g2)
    return log(1/sqrt(2 * π * v) * exp(-(x-m)^2/(2*v)))
end

"""
    logNormRatio(g1,g2)

Computes the log-normalization constant of a division of `g1` with `g2` (the end of the equation ;))

✅ It should be 0 if both rho variables are 0.
# Examples
```julia-repl
julia> logNormRatio(Gaussian1D(0,1) / Gaussian1D(0,0.5))
5.013256549262001
```
"""
function logNormRatio(g1::Gaussian1D, g2::Gaussian1D)
    if (g1.rho == g2.rho) # or else division by zero
        return 0.0;
    end
    x = (g1.tau - g2.tau) / (g1.rho - g2.rho)
    m = g2.tau / g2.rho
    v = 1 / (g1.rho - g2.rho) +  1 / g2.rho
    return log(1/(1/sqrt(2 * π * v) * exp(-(x-m)^2/(2*v))))
end

"""
    show(io,g)

Pretty-prints a 1D Gaussian
"""
function Base.show(io::IO, g::Gaussian1D)
    if (g.rho == 0.0)
        print(io, "μ = 0, σ = Inf")
    else
        print(io, "μ = ", mean(g), ", σ = ", sqrt(variance(g)))
    end
end