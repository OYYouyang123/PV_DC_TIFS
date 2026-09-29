import folium
from folium.features import DivIcon

# ============================================
# 1. 路线坐标数据
#    每条路线 RouteX 有 A~F 六个点
# ============================================
routes = {
    "Route1": [
        ("A", 40.80936, -73.94927),
        ("B", 40.80804, -73.94612),
        ("C", 40.80514, -73.94524),
        ("D", 40.79815, -73.94832),
        ("E", 40.79941, -73.95338),
        ("F", 40.79922, -73.95567),
    ],
    "Route2": [
        ("A", 40.80356, -73.96914),
        ("B", 40.79923, -73.96286),
        ("C", 40.78842, -73.97077),
        ("D", 40.78454, -73.97360),
        ("E", 40.77924, -73.97904),
        ("F", 40.76078, -73.98446),
    ],
    "Route3": [
        ("A", 40.76233, -73.98234),
        ("B", 40.76041, -73.97582),
        ("C", 40.75717, -73.96816),
        ("D", 40.75686, -73.96545),
        ("E", 40.75590, -73.96319),
        ("F", 40.72843, -73.97569),
    ],
    "Route4": [
        ("A", 40.73326, -73.98118),
        ("B", 40.73763, -73.98611),
        ("C", 40.73035, -73.98998),
        ("D", 40.73352, -73.98985),
        ("E", 40.73453, -73.99225),
        ("F", 40.73734, -73.99265),
    ],
    "Route5": [
        ("A", 40.82155, -73.94287),
        ("B", 40.82470, -73.94427),
        ("C", 40.83584, -73.93991),
        ("D", 40.84414, -73.93754),
        ("E", 40.83659, -73.94314),
        ("F", 40.82647, -73.95053),
    ],
    "Route6": [
        ("A", 40.76410, -73.98848),
        ("B", 40.76228, -73.98611),
        ("C", 40.76141, -73.98406),
        ("D", 40.75983, -73.98417),
        ("E", 40.75504, -73.98663),
        ("F", 40.75096, -73.98272),
    ],
}

# ============================================
# 2. 可视化样式设置（不同路线不同颜色）
# ============================================
route_colors = {
    "Route1": "#d1495b",  # 红
    "Route2": "#f4a261",  # 橙
    "Route3": "#5dade2",  # 浅蓝
    "Route4": "#9b59b6",  # 紫
    "Route5": "#1f77b4",  # 蓝
    "Route6": "#2ecc71",  # 绿
}

# ============================================
# 3. 计算地图中心 & 边界
# ============================================
all_lat = [lat for pts in routes.values() for (_, lat, _) in pts]
all_lon = [lon for pts in routes.values() for (_, _, lon) in pts]
center_lat = sum(all_lat) / len(all_lat)
center_lon = sum(all_lon) / len(all_lon)

# ============================================
# 4. 创建地图（使用浅色底图，接近你示意图风格）
# ============================================
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=13,
    tiles="CartoDB Positron",  # 浅灰风格底图（更像示意图）
    control_scale=True
)

# ============================================
# 5. 添加路线：点 + 连线
#    修改内容：
#    1. 连接线粗度改为3
#    2. 点符号颜色与线条颜色对应
#    3. Route1-4使用实心圆圈
#    4. Route5-6使用空心圆圈
#    5. 点圆圈大小调整为4
#    6. 取消A1, A2等标签
# ============================================
for route_name, pts in routes.items():
    color = route_colors.get(route_name, "#333333")

    # 判断路线分组
    route_num = int(route_name.replace("Route", ""))
    is_route1_4 = route_num <= 4

    # 5.1 连线（按 A->F 顺序连接）- 虚线，线粗度改为3
    line_latlon = [(lat, lon) for (_, lat, lon) in pts]
    folium.PolyLine(
        locations=line_latlon,
        color=color,
        weight=3,  # 线粗度改为3
        opacity=0.8,
        tooltip=route_name,
        dash_array='5, 5'  # 虚线样式
    ).add_to(m)

    # 5.2 点 - 使用对应线条颜色
    for letter, lat, lon in pts:
        # 根据路线组设置不同的符号
        if is_route1_4:
            # Route1-4: 实心圆圈
            folium.CircleMarker(
                location=(lat, lon),
                radius=4,  # 圆圈大小调整为4
                color=color,  # 使用对应线条颜色
                weight=1,  # 边框细一点
                fill=True,
                fill_color=color,  # 实心填充，使用对应线条颜色
                fill_opacity=0.9,
                tooltip=f"{route_name}-{letter}",
                popup=f"<b>{route_name}</b><br>Point: {letter}<br>Lat: {lat}<br>Lon: {lon}"
            ).add_to(m)
        else:
            # Route5-6: 空心圆圈
            folium.CircleMarker(
                location=(lat, lon),
                radius=4,  # 圆圈大小调整为4
                color=color,  # 使用对应线条颜色
                weight=2,  # 边框粗一点
                fill=True,
                fill_color="white",  # 白色填充（空心效果）
                fill_opacity=1.0,
                tooltip=f"{route_name}-{letter}",
                popup=f"<b>{route_name}</b><br>Point: {letter}<br>Lat: {lat}<br>Lon: {lon}"
            ).add_to(m)

        # 注意：已取消A1, A2等标签的添加

# ============================================
# 6. 修改图例，添加符号说明
# ============================================
# 创建两种符号的图例项
symbol_legend = """
<div style="display:flex;align-items:center;margin:6px 0;">
    <div style="
        width: 8px;
        height: 8px;
        background-color: #333;
        border-radius: 50%;
        margin-right: 10px;
    "></div>
    <div style="font-size:16px;font-weight:600;color:#333;">Route1-4 (实心圆)</div>
</div>
<div style="display:flex;align-items:center;margin:6px 0;">
    <div style="
        width: 8px;
        height: 8px;
        background-color: white;
        border: 2px solid #333;
        border-radius: 50%;
        margin-right: 10px;
    "></div>
    <div style="font-size:16px;font-weight:600;color:#333;">Route5-6 (空心圆)</div>
</div>
<div style="
    width: 100%;
    height: 1px;
    background-color: #ccc;
    margin: 10px 0;
"></div>
"""

legend_items = ""
for rn in routes.keys():
    c = route_colors[rn]
    legend_items += f"""
    <div style="display:flex;align-items:center;margin:6px 0;">
        <div style="
            width: 34px;
            height: 3px;
            background: {c};
            margin-right: 10px;
            border-radius: 3px;
            border: none;
            border-style: dashed;
            border-width: 0;
        "></div>
        <div style="font-size:16px;font-weight:600;color:#333;">{rn}</div>
    </div>
    """

legend_html = f"""
<div style="
    position: fixed;
    right: 24px;
    bottom: 24px;
    z-index: 9999;
    background: rgba(255,255,255,0.95);
    padding: 16px 18px;
    border-radius: 10px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.15);
    min-width: 180px;
    max-width: 220px;
">
  <div style="font-size:18px;font-weight:800;margin-bottom:10px;color:#333;">Legend</div>
  {symbol_legend}
  <div style="font-size:16px;font-weight:600;margin-bottom:8px;color:#333;">Routes</div>
  {legend_items}
  <div style="
      width: 100%;
      height: 1px;
      background-color: #ccc;
      margin: 10px 0;
  "></div>
  <div style="font-size:14px;font-weight:400;color:#666;margin-top:8px;">
    <div>• 虚线连接 (线粗:3)</div>
    <div>• 实心圆: Route1-4</div>
    <div>• 空心圆: Route5-6</div>
  </div>
</div>
"""

north_html = """
<div style="
    position: fixed;
    right: 30px;
    top: 20px;
    z-index: 9999;
    width: 60px;
    height: 60px;
    text-align:center;
    font-family: Arial, sans-serif;
">
  <div style="font-weight:800;font-size:18px;line-height:18px;color:#333;">N</div>
  <div style="
      margin: 6px auto 0 auto;
      width: 0;
      height: 0;
      border-left: 14px solid transparent;
      border-right: 14px solid transparent;
      border-bottom: 34px solid #333;
  "></div>
</div>
"""

m.get_root().html.add_child(folium.Element(legend_html))
m.get_root().html.add_child(folium.Element(north_html))

# ============================================
# 7. 自适应缩放到所有点范围
# ============================================
min_lat, max_lat = min(all_lat), max(all_lat)
min_lon, max_lon = min(all_lon), max(all_lon)
m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

# ============================================
# 8. 保存
# ============================================
output_file = "routes_schematic_map.html"
m.save(output_file)
print(f"已生成：{output_file}（用浏览器打开）")