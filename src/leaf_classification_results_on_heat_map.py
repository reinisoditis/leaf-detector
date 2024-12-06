import folium
from folium.plugins import HeatMap
from PIL import Image
import piexif
from pathlib import Path
import matplotlib.pyplot as plt
import io
import base64
import json


def add_pie_chart_to_map(map_obj, leaf_conditions):

    # Calculate the total counts for each condition
    condition_counts, _ = calculate_condition_totals(leaf_conditions)

    for image_data in leaf_conditions.values():
        for condition, count in image_data['counts'].items():
            condition_counts[condition] += count

    # Calculate percentages
    total_counts = sum(condition_counts.values())
    percentages = {k: (v / total_counts) * 100 for k, v in condition_counts.items()}

    # Create pie chart
    labels = list(percentages.keys())
    sizes = list(percentages.values())
    colors = ['#2ecc71', '#e74c3c', '#f1c40f']  # Green, Red, Yellow

    fig, ax = plt.subplots(figsize=(3, 3))
    wedges, texts = ax.pie(sizes, startangle=90, colors=colors, autopct=None)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Create legend with percentages
    legend_labels = [f"{label}: {percent:.3f}%" for label, percent in zip(labels, sizes)]
    plt.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(0.5, -0.2), fontsize='small')

    # Save chart to a BytesIO buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)

    # Convert buffer to base64 string
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    img_html = f'<img src="data:image/png;base64,{img_base64}" width="200" height="250">'

    # Add image to the map as an HTML div in the bottom-left corner
    pie_chart_html = f"""
        <div style="position: absolute; bottom: 20px; left: 20px;
                    border:2px solid gray;
                    background-color: white; z-index: 1000;">
            <h4 style="text-align: center; font-size: 14px; margin: 5px;">Summary</h4>
            {img_html}
        </div>
    """
    map_obj.get_root().html.add_child(folium.Element(pie_chart_html))


def calculate_condition_totals(leaf_conditions):
    condition_totals = {'healthy': 0, 'disease': 0, 'mineral': 0}
    all_condition_total = 0

    for image_data in leaf_conditions.values():
        for condition, count in image_data['counts'].items():
            condition_totals[condition] += count
            all_condition_total += count

    return condition_totals, all_condition_total


def extract_gps(image_path):
    """Extract GPS coordinates from image EXIF data"""
    try:
        img = Image.open(image_path)
        exif_dict = piexif.load(img.info['exif'])

        # Check if GPS info is present in image
        gps_info = exif_dict.get('GPS', None)
        if not gps_info:
            print(f"No GPS data found in {image_path}")
            return None

        # Helper function to convert EXIF GPS data to degrees
        def convert_to_degrees(value):
            if isinstance(value, tuple) and len(value) == 3:
                d, m, s = value
                return float(d[0]) / float(d[1]) + (float(m[0]) / float(m[1]) / 60.0) + (
                        float(s[0]) / float(s[1]) / 3600.0)
            return None  # Return None if format is unexpected

        # Extract latitude
        latitude = convert_to_degrees(gps_info.get(2, ()))  # GPSLatitude
        if latitude and gps_info.get(1, b'N') == b'S':  # Check for South latitude
            latitude = -latitude

        # Extract longitude
        longitude = convert_to_degrees(gps_info.get(4, ()))  # GPSLongitude
        if longitude and gps_info.get(3, b'E') == b'W':  # Check for West longitude
            longitude = -longitude

        # Extract altitude
        altitude_data = gps_info.get(6)  # GPSAltitude
        if altitude_data and isinstance(altitude_data, tuple) and len(altitude_data) == 2:
            altitude = float(altitude_data[0]) / float(altitude_data[1])
        else:
            altitude = 0.0  # Default altitude if missing

        return latitude, longitude, altitude

    except Exception as e:
        print(f"Error extracting GPS data from {image_path}: {e}")
        return None


def get_category_for_radius_from_percentage(value):
    if value == 0:
        return 0

    # different heat point radius depending on detected class amount (from, to, radius_size)
    ranges = [
        (60, 100, 20),
        (40, 60, 17),
        (20, 40, 14),
        (10, 20, 11),
        (5, 10, 9),
        (1, 5, 8),
        (0, 1, 5)
    ]

    for lower, upper, result in ranges:
        if lower <= value < upper:
            return result


def create_weighted_heatmap(coordinates, weights, gradient, condition_count, total):
    # Prepare data for heatmap: filter out zero weights (if some classification class was not detected)
    filtered_data = [
        [lat, lon, weight]
        for (lat, lon), weight in zip(coordinates, weights)
        if weight > 0
    ]

    # If there's no valid data after filtering, return None
    if not filtered_data:
        return None

    radius = get_category_for_radius_from_percentage(condition_count/total * 100)

    return HeatMap(
        filtered_data,
        min_opacity=0.3,
        max_opacity=0.8,
        radius=radius,
        blur=4,
        gradient=gradient
    )


def create_map(image_folder, leaf_condition_json, map_output):
    m = None

    # Create feature groups for different condition heatmaps
    healthy_group = folium.FeatureGroup(name='Healthy Leaves')
    disease_group = folium.FeatureGroup(name='Diseased Leaves')
    mineral_group = folium.FeatureGroup(name='Mineral Deficiency')

    # Read data from json file
    with open(leaf_condition_json, 'r') as file:
        leaf_conditions = json.load(file)

    # Store coordinates and weights for each condition
    coordinates = []
    healthy_weights = []
    disease_weights = []
    mineral_weights = []

    # Process all images in test data
    for image_name, data in leaf_conditions.items():
        image_path = Path(image_folder) / image_name
        if not image_path.exists():
            print(f"Warning: Image {image_name} not found in {image_folder}")
            continue

        # Extract GPS coordinates from image
        gps_coordinates = extract_gps(str(image_path))
        if not gps_coordinates:
            print(f"Warning: No GPS data found for {image_name}")
            continue

        lat, lon, altitude = gps_coordinates

        # Initialize map, can pass also max_zoom, to zoom closer, depends on tiles zoom levels
        if m is None:
            m = folium.Map(
                location=[lat, lon],
                zoom_start=18,
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
            )

        coordinates.append((lat, lon))

        # Calculate percentages for each condition
        total_count = sum(data['counts'].values())
        healthy_pct = data['counts'].get('healthy', 0) / total_count if total_count > 0 else 0
        disease_pct = data['counts'].get('disease', 0) / total_count if total_count > 0 else 0
        mineral_pct = data['counts'].get('mineral', 0) / total_count if total_count > 0 else 0

        healthy_weights.append(healthy_pct)
        disease_weights.append(disease_pct)
        mineral_weights.append(mineral_pct)

    if m is None:
        raise ValueError("No valid GPS coordinates found in any images")

    # Heat map colors per classification + color changes (light - dark) per detected condition amount
    healthy_gradient = {
        0.1: 'rgb(229, 255, 229)',  # very light green
        0.2: 'rgb(204, 255, 204)',  # lighter green
        0.4: 'rgb(199, 255, 199)',  # light green
        0.6: 'rgb(118, 255, 118)',  # medium green
        0.8: 'rgb(46, 204, 113)',  # standard green
        1.0: 'rgb(0, 153, 76)'  # dark green
    }

    disease_gradient = {
        0.1: 'rgb(255, 229, 229)',  # very light red
        0.2: 'rgb(255, 204, 204)',  # lighter red
        0.4: 'rgb(255, 199, 199)',  # light red
        0.6: 'rgb(255, 118, 118)',  # medium red
        0.8: 'rgb(231, 76, 60)',  # standard red
        1.0: 'rgb(153, 0, 0)'  # dark red
    }

    mineral_gradient = {
        0.1: 'rgb(255, 250, 229)',  # very light yellow
        0.2: 'rgb(255, 245, 204)',  # lighter yellow
        0.4: 'rgb(255, 242, 199)',  # light yellow
        0.6: 'rgb(255, 230, 118)',  # medium yellow
        0.8: 'rgb(241, 196, 15)',  # standard yellow
        1.0: 'rgb(204, 153, 0)'  # dark yellow
    }

    condition_totals, total_count = calculate_condition_totals(leaf_conditions)

    # Create and add heatmaps for each condition
    if any(healthy_weights):
        healthy_heatmap = create_weighted_heatmap(coordinates, healthy_weights, healthy_gradient, condition_totals['healthy'], total_count)
        healthy_heatmap.add_to(healthy_group)

    if any(disease_weights):
        disease_heatmap = create_weighted_heatmap(coordinates, disease_weights, disease_gradient, condition_totals['disease'], total_count)
        disease_heatmap.add_to(disease_group)

    if any(mineral_weights):
        mineral_heatmap = create_weighted_heatmap(coordinates, mineral_weights, mineral_gradient, condition_totals['mineral'], total_count)
        mineral_heatmap.add_to(mineral_group)

    # Add all groups to map
    healthy_group.add_to(m)
    disease_group.add_to(m)
    mineral_group.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add pie chart to the map
    add_pie_chart_to_map(m, leaf_conditions)

    m.save(str(map_output))
    print(f"Map saved to {map_output}")


# Usage
if __name__ == "__main__":
    image_folder = "folder_with_images_to_process"  # path to image folder, the same as for model pipeline
    leaf_condition_json = "leaf_conditions.json"  # path and .json result file name from model pipeline
    map_output = "heatmap_with_chart_for.html"  # path and .html map name for output
    create_map(image_folder, leaf_condition_json, map_output)
