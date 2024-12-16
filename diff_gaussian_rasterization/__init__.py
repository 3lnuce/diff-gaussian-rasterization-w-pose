#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple

import numpy as np

import torch
import torch.nn as nn

from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [
        item.cpu().clone() if isinstance(item, torch.Tensor) else item
        for item in input_tuple
    ]
    return tuple(copied_tensors)


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    theta,
    rho,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        theta,
        rho,
        raster_settings,
    )


def rasterize_gaussians_fast(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    theta,
    rho,
    raster_settings,
    is_active,
):
    return _RasterizeGaussiansFast.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        theta,
        rho,
        raster_settings,
        is_active,
    )


def rasterize_gaussians_fused(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    theta,
    rho,
    raster_settings,
    is_active,
    ref_color,
    ref_depth,
):
    return _RasterizeGaussiansFused.run(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        theta,
        rho,
        raster_settings,
        is_active,
        ref_color,
        ref_depth,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        theta,
        rho,
        raster_settings,
    ):

        f = open("log_backend_ref", "a+")
        f.write(raster_settings.render_info + "\n")

        tic_loop = torch.cuda.Event(enable_timing=True)
        toc_loop = torch.cuda.Event(enable_timing=True)
        tic_loop.record()

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.projmatrix_raw,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                (
                    num_rendered,
                    color,
                    radii,
                    geomBuffer,
                    binningBuffer,
                    imgBuffer,
                    depth,
                    opacity,
                    n_touched,
                    # out_lambda,
                ) = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print(
                    "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging."
                )
                raise ex
        else:
            (
                num_rendered,
                color,
                radii,
                geomBuffer,
                binningBuffer,
                imgBuffer,
                depth,
                opacity,
                n_touched,
                # out_lambda,
            ) = _C.rasterize_gaussians(*args)

        toc_loop.record()
        torch.cuda.synchronize()
        f.write("[Frontend_ref][rast]: %f\n" % tic_loop.elapsed_time(toc_loop))
        f.flush()

        tic_loop.record()

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            # out_lambda,
        )

        toc_loop.record()
        torch.cuda.synchronize()
        f.write("[Frontend_ref][save]: %f\n" % tic_loop.elapsed_time(toc_loop))
        f.flush()

        return color, radii, depth, opacity, n_touched

    @staticmethod
    def backward(
        ctx,
        grad_out_color,
        grad_out_radii,
        grad_out_depth,
        grad_out_opacity,
        grad_n_touched,
    ):
        # torch.set_printoptions(precision=10, sci_mode=False)
        # print ("grad out color shape: ", grad_out_color.shape)
        # print ("grad out color[0]: ", grad_out_color[0])
        # print ("grad out color[1]: ", grad_out_color[1])
        # print ("grad out color[2]: ", grad_out_color[2])

        # print ("grad out color sum: ", torch.sum(torch.abs(grad_out_color)))
        # print ("zero elements: ", (abs(grad_out_color) != abs(grad_out_color[0,0,0])))
        # print ("sample color data:", grad_out_color[0,0,0], grad_out_color[0,0,1], grad_out_color[0,0,2])
        # print ("grad out depth shape: ", grad_out_depth.shape)
        # print ("grad out depth: ", grad_out_depth)
        # print ("grad out depth sum: ", torch.sum(torch.abs(grad_out_depth)))

        # print ("Reference backward input color shape: ", grad_out_color.shape)
        # print ("Reference backward input color[0]: ", grad_out_color[0])
        # print ("Reference backward input color[0] shape: ", grad_out_color[0].shape)
        # print ("Reference backward input color[0][0]: ")#, grad_out_color[0][0])
        # print ("Reference backward input color[0][0] shape: ", grad_out_color[0][0].shape)

        # print (grad_out_color[0][0])
        # print (grad_out_color[1][0])
        # print (grad_out_color[2][0])
        # print ("Reference backward input color[1][0]: ", grad_out_color[1][0])

        # print ("Reference backward input color[2][0]: ", grad_out_color[2][0])

        # print ("Reference backward input depth: ", grad_out_depth)

        f = open("log_backend_ref", "a+")
        tic_loop = torch.cuda.Event(enable_timing=True)
        toc_loop = torch.cuda.Event(enable_timing=True)
        tic_loop.record()

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            # out_lambda,
        ) = ctx.saved_tensors

        print("===========================================================\n")
        print("[DEBUG]", raster_settings.render_info)
        is_init = False
        if "initialization" in raster_settings.render_info:
            is_init = True
        print(is_init)

        # Restructure args as C++ method expects them
        args = (
            raster_settings.bg,
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.projmatrix_raw,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            grad_out_depth,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            # out_lambda,
            raster_settings.debug,
            is_init,
            raster_settings.render_info,
        )

        toc_loop.record()
        torch.cuda.synchronize()
        f.write("[Backend_grads_ref][load]: %f\n" % tic_loop.elapsed_time(toc_loop))
        f.flush()
        tic_loop.record()

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                (
                    grad_means2D,
                    grad_colors_precomp,
                    grad_opacities,
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_sh,
                    grad_scales,
                    grad_rotations,
                    grad_tau,
                ) = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print(
                    "\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n"
                )
                raise ex
        else:
            (
                grad_means2D,
                grad_colors_precomp,
                grad_opacities,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_sh,
                grad_scales,
                grad_rotations,
                grad_tau,
            ) = _C.rasterize_gaussians_backward(*args)

        toc_loop.record()
        torch.cuda.synchronize()
        f.write("[Backend_grads_ref][rast]: %f\n" % tic_loop.elapsed_time(toc_loop))
        f.flush()
        tic_loop.record()

        grad_tau = torch.sum(grad_tau.view(-1, 6), dim=0)
        grad_rho = grad_tau[:3].view(1, -1)
        grad_theta = grad_tau[3:].view(1, -1)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_theta,
            grad_rho,
            None,
        )

        toc_loop.record()
        torch.cuda.synchronize()
        f.write("[Backend_grads_ref][post]: %f\n" % tic_loop.elapsed_time(toc_loop))
        f.flush()
        # print ("Reference backward gradient: ", grad_means3D)

        return grads


class _RasterizeGaussiansFast(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        theta,
        rho,
        raster_settings,
        is_active,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.projmatrix_raw,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            is_active,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                (
                    num_rendered,
                    color,
                    radii,
                    geomBuffer,
                    binningBuffer,
                    imgBuffer,
                    depth,
                    opacity,
                    n_touched,
                    # out_lambda,
                ) = _C.rasterize_gaussians_fast(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print(
                    "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging."
                )
                raise ex
        else:
            (
                num_rendered,
                color,
                radii,
                geomBuffer,
                binningBuffer,
                imgBuffer,
                depth,
                opacity,
                n_touched,
                # out_lambda,
            ) = _C.rasterize_gaussians_fast(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            # out_lambda,
        )
        return color, radii, depth, opacity, n_touched

    @staticmethod
    def backward(
        ctx,
        grad_out_color,
        grad_out_radii,
        grad_out_depth,
        grad_out_opacity,
        grad_n_touched,
    ):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            # out_lambda,
        ) = ctx.saved_tensors

        print("===========================================================\n")
        print("[DEBUG]", raster_settings.render_info)
        is_init = False
        if "initialization" in raster_settings.render_info:
            is_init = True
        print(is_init)

        f = open("log_backend_bwd", "a+")
        tic_loop = torch.cuda.Event(enable_timing=True)
        toc_loop = torch.cuda.Event(enable_timing=True)
        tic_loop.record()

        # print("debug render info: ", raster_settings.render_info)
        # test = raster_settings.render_info

        # Restructure args as C++ method expects them
        args = (
            raster_settings.bg,
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.projmatrix_raw,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            grad_out_depth,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            # out_lambda,
            raster_settings.debug,
            is_init,
            # test,
            raster_settings.render_info,
        )

        # np.savez_compressed("out_lambda.npz", tensor=out_lambda.detach().cpu().numpy())

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            # TODO: Currently use original backward pass, will change it to cutstomized one later
            try:
                (
                    grad_means2D,
                    grad_colors_precomp,
                    grad_opacities,
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_sh,
                    grad_scales,
                    grad_rotations,
                    grad_tau,
                ) = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print(
                    "\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n"
                )
                raise ex
        else:
            (
                grad_means2D,
                grad_colors_precomp,
                grad_opacities,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_sh,
                grad_scales,
                grad_rotations,
                grad_tau,
            ) = _C.rasterize_gaussians_backward(*args)

        grad_tau = torch.sum(grad_tau.view(-1, 6), dim=0)
        grad_rho = grad_tau[:3].view(1, -1)
        grad_theta = grad_tau[3:].view(1, -1)

        # print("############## before set to zero !!!")

        # filter_scale = (torch.abs(grad_scales) > 1e-5).any(dim=1)
        # filter_rotation = (torch.abs(grad_rotations) > 1e-5).any(dim=1)
        # filter_opacity = (torch.abs(grad_opacities) > 1e-5).any(dim=1)
        # filter_fdc = (torch.abs(grad_cov3Ds_precomp) > 1e-3).any(dim=1)
        # row_indices = filter_scale & filter_rotation & filter_opacity  # & filter_fdc

        # print("indices shape: ", torch.sum(row_indices))

        # grad_means3D[row_indices] = 0
        # grad_means2D[row_indices] = 0
        # grad_sh[row_indices] = 0
        # grad_colors_precomp[row_indices] = 0
        # grad_opacities[row_indices] = 0
        # grad_scales[row_indices] = 0
        # grad_rotations[row_indices] = 0
        # grad_cov3Ds_precomp[row_indices] = 0

        # print(
        #     "################################################## manually set to zero !"
        # )

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_theta,
            grad_rho,
            None,
            None,
        )

        toc_loop.record()
        torch.cuda.synchronize()
        f.write("[Backend_grads_ref][all]: %f\n" % tic_loop.elapsed_time(toc_loop))
        f.flush()

        return grads


class _RasterizeGaussiansFused:
    def run(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        theta,
        rho,
        raster_settings,
        is_active,
        ref_color,
        ref_depth,
    ):

        f = open("log_backend_fused", "a+")
        f.write(raster_settings.render_info + "\n")

        tic_loop = torch.cuda.Event(enable_timing=True)
        toc_loop = torch.cuda.Event(enable_timing=True)
        tic_loop.record()

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.projmatrix_raw,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            is_active,
            ref_color,
            ref_depth,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                (
                    num_rendered,
                    color,
                    radii,
                    geomBuffer,
                    binningBuffer,
                    imgBuffer,
                    depth,
                    opacity,
                    n_touched,
                    grad_means2D,
                    grad_colors_precomp,
                    grad_opacities,
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_sh,
                    grad_scales,
                    grad_rotations,
                    grad_tau,
                ) = _C.rasterize_gaussians_fused(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print(
                    "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging."
                )
                raise ex
        else:
            (
                num_rendered,
                color,
                radii,
                geomBuffer,
                binningBuffer,
                imgBuffer,
                depth,
                opacity,
                n_touched,
                grad_means2D,
                grad_colors_precomp,
                grad_opacities,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_sh,
                grad_scales,
                grad_rotations,
                grad_tau,
            ) = _C.rasterize_gaussians_fused(*args)

        grad_tau = torch.sum(grad_tau.view(-1, 6), dim=0)
        grad_rho = grad_tau[:3].view(1, -1)
        grad_theta = grad_tau[3:].view(1, -1)

        print("grad_means3D.shape: ", grad_means3D.shape)
        print("grad_sh.shape: ", grad_sh.shape)
        print("grad_colors_precomp.shape: ", grad_colors_precomp.shape)
        print("grad_opacities.shape: ", grad_opacities.shape)
        print("grad_scales.shape: ", grad_scales.shape)
        print("grad_rotations.shape: ", grad_rotations.shape)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_theta,
            grad_rho,
            None,
            None,
        )

        toc_loop.record()
        torch.cuda.synchronize()
        f.write("[fused_kernel]: %f\n" % tic_loop.elapsed_time(toc_loop))
        f.flush()
        # return grads

        # Keep relevant tensors for backward
        # ctx.raster_settings = raster_settings
        # ctx.num_rendered = num_rendered
        # ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, depth, opacity, n_touched, grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    projmatrix_raw: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool
    render_info: str


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions, raster_settings.viewmatrix, raster_settings.projmatrix
            )

        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        theta=None,
        rho=None,
    ):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide excatly one of either SHs or precomputed colors!"
            )

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if theta is None:
            theta = torch.Tensor([])
        if rho is None:
            rho = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            theta,
            rho,
            raster_settings,
        )


class GaussianRasterizerFast(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions, raster_settings.viewmatrix, raster_settings.projmatrix
            )

        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        theta=None,
        rho=None,
        is_active=None,
    ):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide excatly one of either SHs or precomputed colors!"
            )

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if theta is None:
            theta = torch.Tensor([])
        if rho is None:
            rho = torch.Tensor([])
        if is_active is None:
            is_active = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians_fast(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            theta,
            rho,
            raster_settings,
            is_active,
        )


class GaussianRasterizerFused:
    def __init__(self, raster_settings):
        # super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions, raster_settings.viewmatrix, raster_settings.projmatrix
            )

        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        theta=None,
        rho=None,
        is_active=None,
        ref_color=None,
        ref_depth=None,
    ):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide excatly one of either SHs or precomputed colors!"
            )

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if theta is None:
            theta = torch.Tensor([])
        if rho is None:
            rho = torch.Tensor([])
        if is_active is None:
            is_active = torch.Tensor([])
        if ref_color is None:
            ref_color = torch.Tensor([])
        if ref_depth is None:
            ref_depth = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians_fused(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            theta,
            rho,
            raster_settings,
            is_active,
            ref_color,
            ref_depth,
        )
