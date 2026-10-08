"""Shaders GLSL del backend GPU (OpenGL 3.3 core)."""

FULLSCREEN_VS = """
#version 330
in vec2 in_pos;
out vec2 uv;
void main() { uv = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

# ---------------------------------------------------------------- formas instanciadas (SDF)
SHAPES_VS = """
#version 330
uniform vec2 u_res;
in vec2 in_pos;
in vec2 in_p0; in vec2 in_p1; in float in_r; in vec4 in_col; in float in_kind;
out vec2 v_pos;
flat out vec2 f_p0; flat out vec2 f_p1; flat out float f_r; flat out vec4 f_col; flat out float f_kind;
void main() {
    int vid = (in_pos.x < 0.0 ? 0 : 2) + (in_pos.y < 0.0 ? 0 : 1);
    vec2 d = in_p1 - in_p0;
    float len = length(d);
    vec2 dir = len > 1e-4 ? d / len : vec2(1.0, 0.0);
    vec2 nrm = vec2(-dir.y, dir.x);
    float pad = in_r + 1.5;
    vec2 a = in_p0 - dir * pad;
    vec2 b = in_p1 + dir * pad;
    vec2 corner = (vid == 0) ? a - nrm * pad : (vid == 1) ? a + nrm * pad : (vid == 2) ? b - nrm * pad : b + nrm * pad;
    v_pos = corner; f_p0 = in_p0; f_p1 = in_p1; f_r = in_r; f_col = in_col; f_kind = in_kind;
    gl_Position = vec4(corner.x / u_res.x * 2.0 - 1.0, corner.y / u_res.y * 2.0 - 1.0, 0.0, 1.0);
}
"""

SHAPES_FS = """
#version 330
in vec2 v_pos;
flat in vec2 f_p0; flat in vec2 f_p1; flat in float f_r; flat in vec4 f_col; flat in float f_kind;
out vec4 frag;
void main() {
    float d;
    if (f_kind < 0.5) {            // cápsula (línea con extremos redondos / disco)
        vec2 pa = v_pos - f_p0, ba = f_p1 - f_p0;
        float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
        d = length(pa - ba * h) - f_r;
    } else if (f_kind < 1.5) {     // segmento con extremos rectos (rectángulo orientado)
        vec2 ba = f_p1 - f_p0;
        float len = length(ba);
        vec2 dir = len > 1e-4 ? ba / len : vec2(1.0, 0.0);
        vec2 pa = v_pos - f_p0;
        float along = dot(pa, dir), across = dot(pa, vec2(-dir.y, dir.x));
        d = max(abs(across) - f_r, max(-along, along - len));
    } else {                       // cuadrado centrado en p0
        vec2 q = abs(v_pos - f_p0);
        d = max(q.x, q.y) - f_r;
    }
    float a = (1.0 - smoothstep(-0.5, 0.5, d)) * f_col.a;
    frag = vec4(f_col.rgb * a, a);
}
"""

# ---------------------------------------------------------------- malla con color por vértice
MESH_VS = """
#version 330
uniform vec2 u_res;
in vec2 in_pos; in vec4 in_col;
out vec4 v_col;
void main() { v_col = in_col; gl_Position = vec4(in_pos.x / u_res.x * 2.0 - 1.0, in_pos.y / u_res.y * 2.0 - 1.0, 0.0, 1.0); }
"""
MESH_FS = """
#version 330
in vec4 v_col; out vec4 frag;
void main() { frag = vec4(v_col.rgb * v_col.a, v_col.a); }
"""

# ---------------------------------------------------------------- sprite (texto / imagen)
SPRITE_VS = """
#version 330
uniform vec2 u_res; uniform vec2 u_center; uniform vec2 u_size; uniform float u_angle;
in vec2 in_pos;
out vec2 uv;
void main() {
    vec2 local = in_pos * u_size * 0.5;
    float c = cos(u_angle), s = sin(u_angle);
    vec2 p = u_center + vec2(local.x * c - local.y * s, local.x * s + local.y * c);
    uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(p.x / u_res.x * 2.0 - 1.0, p.y / u_res.y * 2.0 - 1.0, 0.0, 1.0);
}
"""
SPRITE_FS = """
#version 330
uniform sampler2D tex; uniform float u_alpha; uniform float u_lod;
in vec2 uv; out vec4 frag;
void main() {
    vec4 c = u_lod > 0.01 ? textureLod(tex, uv, u_lod) : texture(tex, uv);
    float a = c.a * u_alpha;
    frag = vec4(c.rgb * a, a);
}
"""

# ---------------------------------------------------------------- fondo (zoom / vibración / brillo, borde espejado)
BACKGROUND_FS = """
#version 330
uniform sampler2D tex; uniform vec2 u_res; uniform float u_zoom; uniform float u_angle; uniform vec2 u_shift; uniform float u_gain;
in vec2 uv; out vec4 frag;
vec2 mirror(vec2 q) { q = mod(q, 2.0); return 1.0 - abs(q - 1.0); }
void main() {
    vec2 p = uv * u_res;
    vec2 c = u_res * 0.5;
    vec2 d = p - c - u_shift;
    float ca = cos(u_angle), sa = sin(u_angle);
    vec2 r = vec2(d.x * ca - d.y * sa, d.x * sa + d.y * ca) / u_zoom + c;
    vec3 col = texture(tex, mirror(r / u_res)).rgb * u_gain;
    frag = vec4(col, 1.0);
}
"""

# ---------------------------------------------------------------- composición de capa / halo
COMPOSITE_FS = """
#version 330
uniform sampler2D tex; uniform float u_opacity;
in vec2 uv; out vec4 frag;
void main() { frag = texture(tex, uv) * u_opacity; }
"""
HALO_FS = """
#version 330
uniform sampler2D tex; uniform float u_k;
in vec2 uv; out vec4 frag;
void main() { vec3 h = texture(tex, uv).rgb * u_k; frag = vec4(h, clamp(max(h.r, max(h.g, h.b)), 0.0, 1.0)); }
"""
COPY_FS = """
#version 330
uniform sampler2D tex;
in vec2 uv; out vec4 frag;
void main() { frag = texture(tex, uv); }
"""
BLUR_FS = """
#version 330
uniform sampler2D tex; uniform vec2 u_step; uniform float u_sigma;
in vec2 uv; out vec4 frag;
void main() {
    int taps = int(min(ceil(u_sigma * 3.0), 40.0));
    float wsum = 1.0;
    vec4 acc = texture(tex, uv);
    for (int i = 1; i <= taps; i++) {
        float w = exp(-0.5 * float(i * i) / (u_sigma * u_sigma));
        acc += (texture(tex, uv + u_step * float(i)) + texture(tex, uv - u_step * float(i))) * w;
        wsum += 2.0 * w;
    }
    frag = acc / wsum;
}
"""
THRESHOLD_FS = """
#version 330
uniform sampler2D tex; uniform float u_thr;
in vec2 uv; out vec4 frag;
void main() { frag = max(texture(tex, uv) - u_thr, 0.0) / max(1.0 - u_thr, 1e-3); }
"""
FINAL_FS = """
#version 330
uniform sampler2D tex; uniform int u_alpha_mode;
in vec2 uv; out vec4 frag;
void main() {
    vec4 c = texture(tex, uv);
    if (u_alpha_mode == 1) {
        vec3 rgb = max(c.rgb, 0.0);
        float a = max(clamp(c.a, 0.0, 1.0), clamp(max(rgb.r, max(rgb.g, rgb.b)), 0.0, 1.0));
        vec3 straight = a > 1e-4 ? rgb / max(a, 1e-4) : vec3(0.0);
        frag = vec4(clamp(straight, 0.0, 1.0), a);
    } else {
        frag = vec4(clamp(c.rgb, 0.0, 1.0), 1.0);
    }
}
"""

# ---------------------------------------------------------------- efectos de post-proceso
_HDR = """
#version 330
uniform sampler2D tex; uniform vec2 u_res; uniform float u_k; uniform int u_alpha;
in vec2 uv; out vec4 frag;
vec2 mirror(vec2 q) { q = mod(q, 2.0); return 1.0 - abs(q - 1.0); }
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
"""

CHROMATIC_FS = _HDR + """
uniform float u_f;
void main() {
    vec2 c = vec2(0.5);
    vec4 g = texture(tex, uv);
    float r = texture(tex, mirror((uv - c) / (1.0 + u_f) + c)).r;
    float b = texture(tex, mirror((uv - c) / max(1.0 - u_f, 0.5) + c)).b;
    frag = vec4(r, g.g, b, g.a);
}
"""
SHAKE_FS = _HDR + """
uniform float u_zoom; uniform float u_angle; uniform vec2 u_shift;
void main() {
    vec2 p = uv * u_res; vec2 c = u_res * 0.5;
    vec2 d = p - c - u_shift;
    float ca = cos(u_angle), sa = sin(u_angle);
    vec2 r = vec2(d.x * ca - d.y * sa, d.x * sa + d.y * ca) / u_zoom + c;
    frag = texture(tex, mirror(r / u_res));
}
"""
VIGNETTE_FS = _HDR + """
uniform float u_strength; uniform float u_soft;
void main() {
    vec2 q = (uv - 0.5) * 2.0;
    float r = length(q) / 1.41421356;
    float m = clamp((r - (1.0 - u_soft)) / u_soft, 0.0, 1.0);
    m = m * m * (3.0 - 2.0 * m);
    frag = texture(tex, uv) * (1.0 - m * u_strength * u_k);
}
"""
COLOR_FS = _HDR + """
uniform float u_hue; uniform float u_sat; uniform float u_contrast; uniform float u_bright; uniform float u_gamma; uniform float u_poster;
vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
void main() {
    vec4 c = texture(tex, uv);
    vec3 rgb = c.rgb;
    if (abs(u_hue) > 1e-3 || abs(u_sat - 1.0) > 1e-4) {
        vec3 hsv = rgb2hsv(clamp(rgb, 0.0, 1.0));
        hsv.x = fract(hsv.x + u_hue / 360.0);
        hsv.y = clamp(hsv.y * (1.0 + (u_sat - 1.0) * u_k), 0.0, 1.0);
        rgb = hsv2rgb(hsv);
    }
    if (abs(u_contrast - 1.0) > 1e-4) rgb = (rgb - 0.5) * (1.0 + (u_contrast - 1.0) * u_k) + 0.5;
    rgb += u_bright * u_k;
    if (abs(u_gamma - 1.0) > 1e-4) rgb = pow(clamp(rgb, 0.0, 1.0), vec3(1.0 / u_gamma));
    if (u_poster > 1.5) rgb = floor(clamp(rgb, 0.0, 1.0) * u_poster) / (u_poster - 1.0);
    frag = vec4(rgb, c.a);
}
"""
PIXELATE_FS = _HDR + """
uniform float u_size;
void main() {
    vec2 p = uv * u_res;
    vec2 q0 = floor(p / u_size) * u_size;
    vec4 acc = vec4(0.0);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            acc += texture(tex, (q0 + vec2(float(i) + 0.5, float(j) + 0.5) * u_size / 3.0) / u_res);
    frag = acc / 9.0;
}
"""
STROBE_FS = _HDR + """
uniform vec3 u_color;
void main() {
    vec4 c = texture(tex, uv);
    vec3 rgb = mix(c.rgb, u_color, u_k);
    float a = u_alpha == 1 ? max(c.a, u_k) : c.a;
    frag = vec4(rgb, a);
}
"""
KALEIDO_FS = _HDR + """
uniform int u_h; uniform int u_v;
void main() {
    vec2 q = uv;
    if (u_h == 1 && q.x > 0.5) q.x = 1.0 - q.x;
    if (u_v == 1 && q.y > 0.5) q.y = 1.0 - q.y;
    frag = mix(texture(tex, uv), texture(tex, q), u_k);
}
"""
RADIAL_FS = _HDR + """
uniform float u_amount; uniform int u_n;
void main() {
    vec2 c = vec2(0.5);
    vec4 acc = texture(tex, uv);
    for (int i = 1; i < u_n; i++) {
        float z = 1.0 + u_amount * u_k * float(i) / float(u_n - 1);
        acc += texture(tex, mirror((uv - c) / z + c));
    }
    frag = acc / float(u_n);
}
"""
SCANLINES_FS = _HDR + """
uniform float u_spacing; uniform float u_dark;
void main() {
    float y = floor(uv.y * u_res.y);
    float row = mod(y, u_spacing) < 0.5 ? 1.0 : 0.0;
    frag = texture(tex, uv) * (1.0 - row * u_dark * u_k);
}
"""
GRAIN_FS = _HDR + """
uniform float u_amount; uniform float u_seed;
void main() {
    vec4 c = texture(tex, uv);
    vec2 cell = floor(uv * u_res * 0.5);
    float n = (hash(cell + u_seed) + hash(cell * 1.7 + u_seed + 3.1) - 1.0) * 1.7;  // aprox. gaussiano
    float g = n * u_amount * u_k;
    if (u_alpha == 1) g *= c.a;
    frag = vec4(c.rgb + g, c.a);
}
"""
GLITCH_FS = _HDR + """
uniform int u_nblocks; uniform vec4 u_blocks[32];   // y0, alto, dx, invertir
uniform vec3 u_rgb; uniform int u_has_rgb;           // dr, db, dy
uniform vec3 u_scan; uniform int u_has_scan;         // espaciado, desfase, fuerza
uniform int u_nnoise; uniform vec4 u_noise[4];       // y0, alto, semilla, fuerza
vec4 blocks(vec2 p) {
    // Mismo orden que la CPU: un bloque con desplazamiento parte de la imagen original (y anula
    // inversiones previas); cada bloque marcado como invertir alterna la inversión.
    float shift = 0.0;
    bool inv = false;
    for (int i = 0; i < u_nblocks; i++) {
        vec4 b = u_blocks[i];
        if (p.y >= b.x && p.y < b.x + b.y) {
            if (abs(b.z) > 0.5) { shift = b.z; inv = false; }
            if (b.w > 0.5) inv = !inv;
        }
    }
    vec2 q = vec2(mod(p.x - shift, u_res.x), p.y);
    vec4 c = texture(tex, (q + 0.5) / u_res);
    if (inv) c = (u_alpha == 1) ? vec4(c.a - c.rgb, c.a) : vec4(1.0 - c.rgb, c.a);
    return c;
}
void main() {
    vec2 p = floor(uv * u_res);
    vec4 c = blocks(p);
    if (u_has_rgb == 1) {
        vec2 pr = vec2(mod(p.x - u_rgb.x, u_res.x), mod(p.y - u_rgb.z, u_res.y));
        vec2 pb = vec2(mod(p.x - u_rgb.y, u_res.x), mod(p.y + u_rgb.z, u_res.y));
        c.r = blocks(pr).r;
        c.b = blocks(pb).b;
    }
    if (u_has_scan == 1 && mod(p.y + u_scan.y, u_scan.x) < 0.5) c *= (1.0 - u_scan.z);
    for (int i = 0; i < u_nnoise; i++) {
        vec4 nb = u_noise[i];
        if (p.y >= nb.x && p.y < nb.x + nb.y) {
            float n = (hash(p + nb.z) * 2.0 - 1.0) * nb.w;
            if (u_alpha == 1) n *= c.a;
            c.rgb += n;
        }
    }
    frag = c;
}
"""
