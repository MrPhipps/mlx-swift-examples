// Copyright © 2024 Apple Inc.

import Foundation
import MLX

// port of https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/sampler.py

/// Interpolate the function defined by `(0 ..< y.count) y)` at positions `xNew`.
func interpolate(y: MLXArray, xNew: MLXArray) -> MLXArray {
    let xLow = xNew.asType(.int32)
    let xHigh = minimum(xLow + 1, y.count - 1)

    let yLow = y[xLow]
    let yHigh = y[xHigh]
    let deltaX = xNew - xLow
    let yNew = yLow * (1 - deltaX) + deltaX * yHigh

    return yNew
}

/// A simple Euler integrator that can be used to sample from our diffusion models.
///
/// The method ``step()`` performs one Euler step from `x_t` to `x_t_prev`.
class SimpleEulerSampler {

    let sigmas: MLXArray
    let alphasCumprod: MLXArray

    public init(configuration: DiffusionConfiguration) {
        let betas: MLXArray

        // compute the noise schedule
        switch configuration.betaSchedule {
        case .linear:
            betas = MLXArray.linspace(
                configuration.betaStart, configuration.betaEnd, count: configuration.trainSteps)
        case .scaledLinear:
            betas = MLXArray.linspace(
                sqrt(configuration.betaStart), sqrt(configuration.betaEnd),
                count: configuration.trainSteps
            ).square()
        }

        let alphas = 1 - betas
        let alphasCumprod = cumprod(alphas)
        self.alphasCumprod = alphasCumprod

        self.sigmas = concatenated([
            MLXArray.zeros([1]), ((1 - alphasCumprod) / alphasCumprod).sqrt(),
        ])
    }

    public var maxTime: Int {
        sigmas.count - 1
    }

    public func samplePrior(shape: [Int], dType: DType = .float32, key: MLXArray? = nil) -> MLXArray
    {
        let noise = MLXRandom.normal(shape, key: key)
        return (noise * sigmas[-1] * (sigmas[-1].square() + 1).rsqrt()).asType(dType)
    }

    public func addNoise(x: MLXArray, t: MLXArray, key: MLXArray? = nil) -> MLXArray {
        let noise = MLXRandom.normal(x.shape, key: key)
        let s = sigmas(t)
        return (x + noise * s) * (s.square() + 1).rsqrt()
    }

    public func sigmas(_ t: MLXArray) -> MLXArray {
        interpolate(y: sigmas, xNew: t)
    }

    public func timeSteps(steps: Int, start: Int? = nil, dType: DType = .float32) -> [(
        MLXArray, MLXArray
    )] {
        let start = start ?? (sigmas.count - 1)
        precondition(0 < start)
        precondition(start <= sigmas.count - 1)
        let steps = MLX.linspace(start, 0, count: steps + 1).asType(dType)

        return Array(zip(steps, steps[1...]))
    }

    open func step(epsPred: MLXArray, xt: MLXArray, t: MLXArray, tPrev: MLXArray) -> MLXArray {
        let dtype = epsPred.dtype
        let sigma = sigmas(t).asType(dtype)
        let sigmaPrev = sigmas(tPrev).asType(dtype)

        let dt = sigmaPrev - sigma
        var xtPrev = (sigma.square() + 1).sqrt() * xt + epsPred * dt
        xtPrev = xtPrev * (sigmaPrev.square() + 1).rsqrt()

        return xtPrev
    }
}

class SimpleEulerAncestralSampler: SimpleEulerSampler {

    open override func step(epsPred: MLXArray, xt: MLXArray, t: MLXArray, tPrev: MLXArray)
        -> MLXArray
    {
        let dtype = epsPred.dtype
        let sigma = sigmas(t).asType(dtype)
        let sigmaPrev = sigmas(tPrev).asType(dtype)

        let sigma2 = sigma.square()
        let sigmaPrev2 = sigmaPrev.square()
        let sigmaUp = (sigmaPrev2 * (sigma2 - sigmaPrev2) / sigma2).sqrt()
        let sigmaDown = (sigmaPrev2 - sigmaUp ** 2).sqrt()

        let dt = sigmaDown - sigma
        var xtPrev = (sigma2 + 1).sqrt() * xt + epsPred * dt
        let noise = MLXRandom.normal(xtPrev.shape).asType(xtPrev.dtype)
        xtPrev = xtPrev + noise * sigmaUp
        xtPrev = xtPrev * (sigmaPrev2 + 1).rsqrt()

        return xtPrev
    }
}

/// CompVis `make_ddim_timesteps(ddim_discr_method="uniform")`: `range(0, T, T / steps) + 1`,
/// returned in descending order. Yields one extra step when `T` isn't divisible by
/// `steps` (CompVis does too).
func compVisDDIMTimesteps(steps: Int, trainSteps: Int) -> [Int] {
    let c = max(trainSteps / max(steps, 1), 1)
    return Array(stride(from: 0, to: trainSteps, by: c).map { $0 + 1 }.reversed())
}

/// DDIM (Song et al. 2020) as implemented by CompVis `ldm.models.diffusion.ddim.DDIMSampler`,
/// working directly in the variance-preserving `x_t` space the UNet sees. Timesteps are
/// real training timesteps (passed to the UNet unshifted); `tPrev == 0` on the final
/// step resolves to `alphas_cumprod[0]`, as CompVis's `ddim_alphas_prev` does.
class DDIMSampler: SimpleEulerSampler {
    let eta: Float

    init(configuration: DiffusionConfiguration, eta: Float) {
        self.eta = eta
        super.init(configuration: configuration)
    }

    override var maxTime: Int {
        alphasCumprod.count - 1
    }

    /// Unit-variance Gaussian, like CompVis's `torch.randn` start code.
    override func samplePrior(shape: [Int], dType: DType = .float32, key: MLXArray? = nil)
        -> MLXArray
    {
        MLXRandom.normal(shape, key: key).asType(dType)
    }

    override func timeSteps(steps: Int, start: Int? = nil, dType: DType = .float32) -> [(
        MLXArray, MLXArray
    )] {
        let ts = compVisDDIMTimesteps(steps: steps, trainSteps: alphasCumprod.count)
        let prevs = Array(ts.dropFirst()) + [0]
        return zip(ts, prevs).map { (MLXArray(Float($0)).asType(dType), MLXArray(Float($1)).asType(dType)) }
    }

    func alphaBar(_ t: MLXArray) -> MLXArray {
        alphasCumprod[t.asType(.int32)]
    }

    /// The deterministic part of a DDIM update, plus `sigma` for the stochastic part.
    func ddimUpdate(epsPred: MLXArray, xt: MLXArray, t: MLXArray, tPrev: MLXArray, eta: Float)
        -> (mean: MLXArray, sigma: MLXArray)
    {
        let a = alphaBar(t)
        let aPrev = alphaBar(tPrev)
        let sigma = eta * ((1 - aPrev) / (1 - a) * (1 - a / aPrev)).sqrt()
        let predX0 = (xt - (1 - a).sqrt() * epsPred) / a.sqrt()
        let direction = (1 - aPrev - sigma.square()).sqrt() * epsPred
        return (aPrev.sqrt() * predX0 + direction, sigma)
    }

    override func step(epsPred: MLXArray, xt: MLXArray, t: MLXArray, tPrev: MLXArray) -> MLXArray {
        let dtype = epsPred.dtype
        let computeType = DType.float32
        let (mean, sigma) = ddimUpdate(
            epsPred: epsPred.asType(computeType), xt: xt.asType(computeType), t: t, tPrev: tPrev,
            eta: eta)
        guard eta > 0 else {
            return mean.asType(dtype)
        }
        let noise = MLXRandom.normal(xt.shape)
        return (mean + sigma * noise).asType(dtype)
    }
}

/// PLMS (Liu et al. 2022) as implemented by CompVis `ldm.models.diffusion.plms.PLMSSampler`.
/// The first step needs a second UNet evaluation at the next timestep; the generic
/// denoise loop provides it by visiting an extra "correction" step right after the
/// first, which this sampler recognises by call order. One instance per run.
final class PLMSSampler: DDIMSampler {
    private var oldEps: [MLXArray] = []
    private var firstStep: (eps: MLXArray, xt: MLXArray, t: MLXArray, tPrev: MLXArray)?
    private var calls = 0

    init(configuration: DiffusionConfiguration) {
        super.init(configuration: configuration, eta: 0)
    }

    override func timeSteps(steps: Int, start: Int? = nil, dType: DType = .float32) -> [(
        MLXArray, MLXArray
    )] {
        let pairs = super.timeSteps(steps: steps, start: start, dType: dType)
        guard let first = pairs.first else { return pairs }
        // Correction visit: the UNet runs at CompVis's `t_next` (the next timestep,
        // or the same one for a single-step run) on the provisional x_{t_next}.
        let tNext = pairs.count > 1 ? pairs[1].0 : first.0
        return [first, (tNext, first.1)] + pairs.dropFirst()
    }

    override func step(epsPred: MLXArray, xt: MLXArray, t: MLXArray, tPrev: MLXArray) -> MLXArray {
        let dtype = epsPred.dtype
        let eps = epsPred.asType(.float32)
        let x = xt.asType(.float32)
        defer { calls += 1 }

        switch calls {
        case 0:
            firstStep = (eps, x, t, tPrev)
            return ddimUpdate(epsPred: eps, xt: x, t: t, tPrev: tPrev, eta: 0).mean.asType(dtype)
        case 1:
            guard let first = firstStep else { fatalError("PLMS correction step without a first step") }
            let ePrime = (first.eps + eps) / 2
            oldEps.append(first.eps)
            return ddimUpdate(epsPred: ePrime, xt: first.xt, t: first.t, tPrev: first.tPrev, eta: 0)
                .mean.asType(dtype)
        default:
            let ePrime: MLXArray
            switch oldEps.count {
            case 1:
                ePrime = (3 * eps - oldEps[0]) / 2
            case 2:
                ePrime = (23 * eps - 16 * oldEps[1] + 5 * oldEps[0]) / 12
            default:
                let n = oldEps.count
                ePrime = (55 * eps - 59 * oldEps[n - 1] + 37 * oldEps[n - 2] - 9 * oldEps[n - 3]) / 24
            }
            oldEps.append(eps)
            if oldEps.count > 3 { oldEps.removeFirst() }
            return ddimUpdate(epsPred: ePrime, xt: x, t: t, tPrev: tPrev, eta: 0).mean.asType(dtype)
        }
    }
}

