import taichi as ti

# 初始化，不带 arch 参数则自动选择
ti.init() 

# 屏幕分辨率
res_x, res_y = 1200, 1200
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# --- 交互参数定义 ---
ka = ti.field(dtype=ti.f32, shape=())
kd = ti.field(dtype=ti.f32, shape=())
ks = ti.field(dtype=ti.f32, shape=())
shininess = ti.field(dtype=ti.f32, shape=())

# 初始化默认值
ka[None] = 0.2 #环境光
kd[None] = 0.7 #漫反射
ks[None] = 0.5 #镜面高光强度
shininess[None] = 32.0 #高光锐度

# --- 场景常量定义 ---
camera_pos = ti.Vector([0.0, 0.0, 5.0]) #摄像头位置
light_pos = ti.Vector([2.0, 3.0, 4.0]) #光源位置
light_color = ti.Vector([1.0, 1.0, 1.0])
bg_color = ti.Vector([0.0, 0.1, 0.1])  # 深青色背景

# 红色球体
sphere_center = ti.Vector([-1.2, -0.2, 0.0])
sphere_radius = 1.2
sphere_color = ti.Vector([0.8, 0.1, 0.1])

# 紫色圆锥
cone_tip = ti.Vector([1.2, 1.2, 0.0])
cone_height = 2.6
cone_radius = 1.2
cone_color = ti.Vector([0.6, 0.2, 0.8])

@ti.func
def intersect_sphere(ray_o, ray_d):
    oc = ray_o - sphere_center
    a = ray_d.dot(ray_d)
    b = 2.0 * oc.dot(ray_d)
    c = oc.dot(oc) - sphere_radius**2
    discriminant = b**2 - 4 * a * c
    t = -1.0
    if discriminant > 0:
        t = (-b - ti.sqrt(discriminant)) / (2.0 * a)
    return t

@ti.func
def intersect_cone(ray_o, ray_d):
    k = cone_radius / cone_height
    m = k**2
    co = ray_o - cone_tip
    a = ray_d.x**2 + ray_d.z**2 - m * ray_d.y**2
    b = 2.0 * (ray_d.x * co.x + ray_d.z * co.z - m * ray_d.y * co.y)
    c = co.x**2 + co.z**2 - m * co.y**2
    discriminant = b**2 - 4 * a * c
    t_res = -1.0
    if discriminant > 0:
        t1 = (-b - ti.sqrt(discriminant)) / (2.0 * a)
        t2 = (-b + ti.sqrt(discriminant)) / (2.0 * a)
        if t1 > 0:
            y1 = ray_o.y + t1 * ray_d.y
            if (cone_tip.y - cone_height) < y1 < cone_tip.y:
                t_res = t1
        if t2 > 0:
            y2 = ray_o.y + t2 * ray_d.y
            if (cone_tip.y - cone_height) < y2 < cone_tip.y:
                if t_res < 0 or t2 < t_res:
                    t_res = t2
    return t_res

@ti.func
def get_sphere_normal(p):
    return (p - sphere_center).normalized()

@ti.func
def get_cone_normal(p):
    cp = p - cone_tip
    r_side = ti.Vector([cp.x, 0.0, cp.z]).normalized()
    k = cone_radius / cone_height
    return ti.Vector([r_side.x, k, r_side.z]).normalized()

@ti.kernel #核心渲染函数
def render():
    for i, j in pixels:
        uv_x = (i / res_x) * 2.0 - 1.0
        uv_y = (j / res_y) * 2.0 - 1.0
        ray_dir = ti.Vector([uv_x, uv_y, -1.0]).normalized()

        t_sphere = intersect_sphere(camera_pos, ray_dir)
        t_cone = intersect_cone(camera_pos, ray_dir)

        t_min = -1.0
        obj_id = 0

        if t_sphere > 0:
            t_min = t_sphere
            obj_id = 1
        if t_cone > 0 and (t_min < 0 or t_cone < t_min):
            t_min = t_cone
            obj_id = 2

        if obj_id == 0:
            pixels[i, j] = bg_color
        else:
            hit_pos = camera_pos + t_min * ray_dir
            n = ti.Vector([0.0, 0.0, 0.0])
            obj_c = ti.Vector([0.0, 0.0, 0.0])

            if obj_id == 1:
                n = get_sphere_normal(hit_pos)
                obj_c = sphere_color
            else:
                n = get_cone_normal(hit_pos)
                obj_c = cone_color

            v = (camera_pos - hit_pos).normalized()  # 视线向量
            l = (light_pos - hit_pos).normalized()  # 光照向量

            # --- 2. 硬阴影计算 (Hard Shadow) ---
            # 引入微小的偏移 eps 防止“自遮挡”带来的黑点噪声
            eps = 1e-4
            shadow_ray_o = hit_pos + n * eps
            shadow_ray_d = l

            dist_to_light = (light_pos - hit_pos).norm()
            in_shadow = False

            # 检测阴影射线是否击中其他物体
            t_s = intersect_sphere(shadow_ray_o, shadow_ray_d)
            if 0 < t_s < dist_to_light:
                in_shadow = True

            t_c = intersect_cone(shadow_ray_o, shadow_ray_d)
            if 0 < t_c < dist_to_light:
                in_shadow = True

            # --- 1. Blinn-Phong 模型计算 ---
            # 环境光分量始终存在
            amb = ka[None] * light_color * obj_c

            diff = ti.Vector([0.0, 0.0, 0.0])
            spec = ti.Vector([0.0, 0.0, 0.0])

            if not in_shadow:
                # 漫反射
                diff = kd[None] * ti.max(n.dot(l), 0.0) * light_color * obj_c

                # Blinn-Phong: 使用半程向量 H
                h = (v + l).normalized()
                spec = ks[None] * ti.pow(ti.max(n.dot(h), 0.0), shininess[None]) * light_color

            pixels[i, j] = amb + diff + spec
            
def main():
    window = ti.ui.Window("Phong Rendering Lab", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    while window.running:
        render()
        canvas.set_image(pixels)
        
        with gui.sub_window("Material Controls", 0.05, 0.05, 0.3, 0.15):
            # 注意：slider_float 返回拖动后的新数值
            ka[None] = gui.slider_float("Ka (Ambient)", ka[None], 0.0, 1.0)
            kd[None] = gui.slider_float("Kd (Diffuse)", kd[None], 0.0, 1.0)
            ks[None] = gui.slider_float("Ks (Specular)", ks[None], 0.0, 1.0)
            shininess[None] = gui.slider_float("Shininess", shininess[None], 1.0, 128.0)
            
        window.show()

if __name__ == "__main__":
    main()