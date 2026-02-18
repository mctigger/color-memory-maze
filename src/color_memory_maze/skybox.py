from dm_control import mjcf
from dm_control.locomotion.arenas import labmaze_textures
from dm_control.locomotion.arenas.labmaze_textures import labmaze_assets


class FixedSkyBox(labmaze_textures.SkyBox):
    """Represents a texture asset for the sky box."""

    def _build(self, style):
        labmaze_textures = labmaze_assets.get_sky_texture_paths("sky_03")
        left = labmaze_textures.left
        right = labmaze_textures.right
        up = labmaze_textures.up
        down = labmaze_textures.down
        back = labmaze_textures.back
        front = labmaze_textures.front
        self._mjcf_root = mjcf.RootElement(model="labmaze_" + style)
        self._texture = self._mjcf_root.asset.add(
            "texture",
            type="skybox",
            name="texture",
            fileleft=left,
            fileright=right,
            fileup=up,
            filedown=down,
            filefront=front,
            fileback=back,
        )

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def texture(self):
        return self._texture
