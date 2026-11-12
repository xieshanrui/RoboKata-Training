"""Apply the recorded raster adaptation to a pinned official benchmark checkout.

Default is read-only. This is not an installer and never changes the RoboICL
collection checkout: its base-task hash differs and is explicitly rejected.
"""
import argparse
import hashlib
import json
from pathlib import Path


BASE_SHA = "448f7152b65cb9102217cf5463aa821d72810ca56f63d5a797ec7bd43e23e101"
RASTER_SHA = "595206aba65c1626f19efd26c543ced2da0b5e214eb141d631f3f819d61a2d1e"
TASK_SHA = {
    "move_stapler_pad": "c2b7aa7bf4e896afc392a8cf1ac107b6982088a459c716de005a3412fcdedd6d",
    "stamp_seal": "28a159e27aa76459355fa1f81e962815c62ee6435727039d73664b7eb6731b82",
}
OLD = '''        set_global_config(max_num_materials=50000, max_num_textures=50000)
        self.renderer = sapien.SapienRenderer()
        # give renderer to sapien sim
        self.engine.set_renderer(self.renderer)

        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")

        # declare sapien scene
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
'''
NEW = '''        set_global_config(max_num_materials=50000, max_num_textures=50000)
        render_system = sapien.render.RenderSystem(device="pci:" + os.environ["ROBOICL_RENDER_PCI"])
        self.renderer = sapien.SapienRenderer()
        # give renderer to sapien sim
        self.engine.set_renderer(self.renderer)

        sapien.render.set_camera_shader_dir("default")

        # declare sapien scene
        scene_config = sapien.SceneConfig()
        sapien.physx.set_scene_config(scene_config)
        self.scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), render_system])
        assert self.scene.render_system.device.pci_string.lower() == os.environ["ROBOICL_RENDER_PCI"].lower(), "Renderer GPU mismatch"
'''


def digest(text):
    # Git checkouts may have CRLF; compare canonical UTF-8 source.
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def plan(root):
    root = root.resolve(strict=True)
    base = root / "envs/_base_task.py"
    text = base.read_text(encoding="utf-8")
    before = digest(text)
    if before not in (BASE_SHA, RASTER_SHA):
        raise ValueError("Expected official RoboTwin 2eeec322 base task; refusing to patch a different implementation")
    if before == BASE_SHA:
        if text.count(OLD) != 1:
            raise ValueError("Pinned renderer block missing")
        text = text.replace(OLD, NEW)
    if digest(text) != RASTER_SHA:
        raise ValueError("Raster patch does not match the recorded runtime")
    changes = [(base, text)]
    upstream = Path(__file__).resolve().parents[2] / "evaluation/robotwin/envs"
    for name, original in TASK_SHA.items():
        path = root / "envs" / f"{name}.py"
        override = (upstream / f"{name}.py").read_text(encoding="utf-8")
        if digest(path.read_text(encoding="utf-8")) not in (original, digest(override)):
            raise ValueError(f"Unrecognized task implementation: {name}")
        changes.append((path, override))
    # Validate every target before the first write.
    for path, _ in changes:
        if not path.resolve(strict=True).is_relative_to(root):
            raise ValueError(f"Target escapes checkout: {path}")
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robotwin", type=Path)
    parser.add_argument("--apply", action="store_true", help="Apply after checking every source identity")
    args = parser.parse_args()
    changes = plan(args.robotwin)
    if args.apply:
        for path, text in changes:
            path.write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps({"applied": args.apply, "files": {str(p): digest(s) for p, s in changes}}, indent=2))


if __name__ == "__main__":
    main()
