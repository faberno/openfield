from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _positive_angle(angle):
    two_pi = 2.0 * math.pi
    if angle < 0.0:
        angle += two_pi
    if angle >= two_pi:
        angle -= two_pi
    return angle


@njit(cache=True)
def _point_in_polygon_2d(vertices_2d, count, x, y):
    inside = False
    previous_x = vertices_2d[count - 1, 0]
    previous_y = vertices_2d[count - 1, 1]
    for index in range(count):
        current_x = vertices_2d[index, 0]
        current_y = vertices_2d[index, 1]
        edge_x = current_x - previous_x
        edge_y = current_y - previous_y
        rel_x = x - previous_x
        rel_y = y - previous_y
        cross = edge_x * rel_y - edge_y * rel_x
        if (
            abs(cross) <= 1e-12
            and min(previous_x, current_x) - 1e-12 <= x <= max(previous_x, current_x) + 1e-12
            and min(previous_y, current_y) - 1e-12 <= y <= max(previous_y, current_y) + 1e-12
        ):
            return True
        if (previous_y > y) != (current_y > y):
            intersection_x = (current_x - previous_x) * (y - previous_y) / (current_y - previous_y) + previous_x
            if x < intersection_x:
                inside = not inside
        previous_x = current_x
        previous_y = current_y
    return inside


@njit(cache=True)
def _sort_unique_positive(values, count):
    for index in range(1, count):
        value = values[index]
        position = index - 1
        while position >= 0 and values[position] > value:
            values[position + 1] = values[position]
            position -= 1
        values[position + 1] = value

    unique_count = 0
    for index in range(count):
        value = values[index]
        if value < 0.0:
            value = 0.0
        if unique_count == 0 or value - values[unique_count - 1] > 1e-15:
            values[unique_count] = value
            unique_count += 1
    return unique_count


@njit(cache=True)
def prepare_flat_facet(point, vertices, normal, sound_speed):
    vertex_count = vertices.shape[0]
    prepared_normal = np.empty(3, dtype=np.float64)
    centroid = np.zeros(3, dtype=np.float64)
    vertices_2d = np.empty((vertex_count, 2), dtype=np.float64)
    critical_radii = np.empty(2 * vertex_count + 1, dtype=np.float64)
    normal_angles = np.empty(vertex_count, dtype=np.float64)
    thresholds = np.empty(vertex_count, dtype=np.float64)

    normal_norm = math.sqrt(
        normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]
    )
    if normal_norm <= 1e-18:
        return (
            False,
            prepared_normal,
            centroid,
            0.0,
            vertices_2d,
            critical_radii,
            0,
            normal_angles,
            thresholds,
            0,
            False,
            0.0,
            0.0,
        )
    for axis in range(3):
        prepared_normal[axis] = normal[axis] / normal_norm

    tangent_x = np.empty(3, dtype=np.float64)
    for axis in range(3):
        tangent_x[axis] = vertices[1, axis] - vertices[0, axis]
    tangent_dot = (
        tangent_x[0] * prepared_normal[0]
        + tangent_x[1] * prepared_normal[1]
        + tangent_x[2] * prepared_normal[2]
    )
    for axis in range(3):
        tangent_x[axis] -= tangent_dot * prepared_normal[axis]
    tangent_x_norm = math.sqrt(
        tangent_x[0] * tangent_x[0] + tangent_x[1] * tangent_x[1] + tangent_x[2] * tangent_x[2]
    )
    if tangent_x_norm <= 1e-18:
        return (
            False,
            prepared_normal,
            centroid,
            0.0,
            vertices_2d,
            critical_radii,
            0,
            normal_angles,
            thresholds,
            0,
            False,
            0.0,
            0.0,
        )
    for axis in range(3):
        tangent_x[axis] /= tangent_x_norm

    tangent_y = np.empty(3, dtype=np.float64)
    tangent_y[0] = prepared_normal[1] * tangent_x[2] - prepared_normal[2] * tangent_x[1]
    tangent_y[1] = prepared_normal[2] * tangent_x[0] - prepared_normal[0] * tangent_x[2]
    tangent_y[2] = prepared_normal[0] * tangent_x[1] - prepared_normal[1] * tangent_x[0]

    rel0 = point[0] - vertices[0, 0]
    rel1 = point[1] - vertices[0, 1]
    rel2 = point[2] - vertices[0, 2]
    signed_distance = rel0 * prepared_normal[0] + rel1 * prepared_normal[1] + rel2 * prepared_normal[2]
    distance_to_plane = abs(signed_distance)
    projected_point = np.empty(3, dtype=np.float64)
    for axis in range(3):
        projected_point[axis] = point[axis] - signed_distance * prepared_normal[axis]

    for vertex_index in range(vertex_count):
        rel_x = vertices[vertex_index, 0] - projected_point[0]
        rel_y = vertices[vertex_index, 1] - projected_point[1]
        rel_z = vertices[vertex_index, 2] - projected_point[2]
        vertices_2d[vertex_index, 0] = rel_x * tangent_x[0] + rel_y * tangent_x[1] + rel_z * tangent_x[2]
        vertices_2d[vertex_index, 1] = rel_x * tangent_y[0] + rel_y * tangent_y[1] + rel_z * tangent_y[2]
        centroid[0] += vertices[vertex_index, 0]
        centroid[1] += vertices[vertex_index, 1]
        centroid[2] += vertices[vertex_index, 2]
    centroid[0] /= vertex_count
    centroid[1] /= vertex_count
    centroid[2] /= vertex_count

    critical_count = 0
    origin_inside = _point_in_polygon_2d(vertices_2d, vertex_count, 0.0, 0.0)
    if origin_inside:
        critical_radii[critical_count] = 0.0
        critical_count += 1

    for vertex_index in range(vertex_count):
        x = vertices_2d[vertex_index, 0]
        y = vertices_2d[vertex_index, 1]
        critical_radii[critical_count] = math.sqrt(x * x + y * y)
        critical_count += 1

    for vertex_index in range(vertex_count):
        start_x = vertices_2d[vertex_index, 0]
        start_y = vertices_2d[vertex_index, 1]
        next_index = (vertex_index + 1) % vertex_count
        segment_x = vertices_2d[next_index, 0] - start_x
        segment_y = vertices_2d[next_index, 1] - start_y
        length_squared = segment_x * segment_x + segment_y * segment_y
        if length_squared <= 1e-30:
            continue
        fraction = -(start_x * segment_x + start_y * segment_y) / length_squared
        if -1e-12 <= fraction <= 1.0 + 1e-12:
            if fraction < 0.0:
                fraction = 0.0
            elif fraction > 1.0:
                fraction = 1.0
            closest_x = start_x + fraction * segment_x
            closest_y = start_y + fraction * segment_y
            critical_radii[critical_count] = math.sqrt(closest_x * closest_x + closest_y * closest_y)
            critical_count += 1

    critical_count = _sort_unique_positive(critical_radii, critical_count)

    area2 = 0.0
    for vertex_index in range(vertex_count):
        next_index = (vertex_index + 1) % vertex_count
        area2 += (
            vertices_2d[vertex_index, 0] * vertices_2d[next_index, 1]
            - vertices_2d[vertex_index, 1] * vertices_2d[next_index, 0]
        )
    angular_count = 0
    if abs(area2) > 1e-30:
        orientation = 1.0 if area2 > 0.0 else -1.0
        convex = True
        for vertex_index in range(vertex_count):
            previous_index = (vertex_index + vertex_count - 1) % vertex_count
            next_index = (vertex_index + 1) % vertex_count
            first_x = vertices_2d[vertex_index, 0] - vertices_2d[previous_index, 0]
            first_y = vertices_2d[vertex_index, 1] - vertices_2d[previous_index, 1]
            second_x = vertices_2d[next_index, 0] - vertices_2d[vertex_index, 0]
            second_y = vertices_2d[next_index, 1] - vertices_2d[vertex_index, 1]
            cross = first_x * second_y - first_y * second_x
            if orientation * cross < -1e-12:
                convex = False
                break
        if convex:
            for vertex_index in range(vertex_count):
                next_index = (vertex_index + 1) % vertex_count
                edge_x = vertices_2d[next_index, 0] - vertices_2d[vertex_index, 0]
                edge_y = vertices_2d[next_index, 1] - vertices_2d[vertex_index, 1]
                edge_norm = math.sqrt(edge_x * edge_x + edge_y * edge_y)
                if edge_norm <= 1e-30:
                    continue
                normal_x = -orientation * edge_y
                normal_y = orientation * edge_x
                threshold = orientation * (
                    edge_x * vertices_2d[vertex_index, 1]
                    - edge_y * vertices_2d[vertex_index, 0]
                )
                normal_angles[angular_count] = _positive_angle(math.atan2(normal_y, normal_x))
                thresholds[angular_count] = threshold / edge_norm
                angular_count += 1

    support_start = math.sqrt(
        distance_to_plane * distance_to_plane + critical_radii[0] * critical_radii[0]
    ) / sound_speed
    support_end = math.sqrt(
        distance_to_plane * distance_to_plane
        + critical_radii[critical_count - 1] * critical_radii[critical_count - 1]
    ) / sound_speed
    return (
        True,
        prepared_normal,
        centroid,
        distance_to_plane,
        vertices_2d,
        critical_radii,
        critical_count,
        normal_angles,
        thresholds,
        angular_count,
        origin_inside,
        support_start,
        support_end,
    )


@njit(cache=True)
def _convex_polygon_angular_measure_count(
    normal_angles,
    thresholds,
    edge_total,
    origin_inside,
    radius,
    lefts,
    rights,
    new_lefts,
    new_rights,
    edge_lefts,
    edge_rights,
):
    two_pi = 2.0 * math.pi
    if radius <= 1e-18:
        return two_pi if origin_inside else 0.0

    interval_count = 1
    lefts[0] = 0.0
    rights[0] = two_pi

    for edge_index in range(edge_total):
        q = thresholds[edge_index] / radius
        if q >= 1.0:
            if q > 1.0 + 1e-12:
                return 0.0
            edge_count = 1
            edge_lefts[0] = normal_angles[edge_index]
            edge_rights[0] = normal_angles[edge_index]
        elif q <= -1.0:
            continue
        else:
            center = normal_angles[edge_index]
            delta = math.acos(q)
            left = _positive_angle(center - delta)
            right = _positive_angle(center + delta)
            if left <= right:
                edge_count = 1
                edge_lefts[0] = left
                edge_rights[0] = right
            else:
                edge_count = 2
                edge_lefts[0] = left
                edge_rights[0] = two_pi
                edge_lefts[1] = 0.0
                edge_rights[1] = right

        new_count = 0
        for interval_index in range(interval_count):
            for edge_interval_index in range(edge_count):
                left = lefts[interval_index]
                if edge_lefts[edge_interval_index] > left:
                    left = edge_lefts[edge_interval_index]
                right = rights[interval_index]
                if edge_rights[edge_interval_index] < right:
                    right = edge_rights[edge_interval_index]
                if right - left > 1e-14 and new_count < 8:
                    new_lefts[new_count] = left
                    new_rights[new_count] = right
                    new_count += 1

        if new_count == 0:
            return 0.0
        interval_count = new_count
        for interval_index in range(interval_count):
            lefts[interval_index] = new_lefts[interval_index]
            rights[interval_index] = new_rights[interval_index]

    measure = 0.0
    for interval_index in range(interval_count):
        measure += rights[interval_index] - lefts[interval_index]
    return measure


@njit(cache=True)
def _convex_polygon_angular_measure(
    normal_angles,
    thresholds,
    origin_inside,
    radius,
    lefts,
    rights,
    new_lefts,
    new_rights,
    edge_lefts,
    edge_rights,
):
    return _convex_polygon_angular_measure_count(
        normal_angles,
        thresholds,
        normal_angles.shape[0],
        origin_inside,
        radius,
        lefts,
        rights,
        new_lefts,
        new_rights,
        edge_lefts,
        edge_rights,
    )


@njit(cache=True)
def add_exact_flat_polygon_response(
    output,
    normal_angles,
    thresholds,
    origin_inside,
    critical_radii,
    distance_to_plane,
    first,
    last,
    start_time,
    delay,
    propagation_delay,
    weight_scale,
    baffle_distance,
    has_baffle,
    sound_speed,
    sampling_frequency,
    nodes,
    weights,
):
    dt = 1.0 / sampling_frequency
    event_times = np.empty(critical_radii.shape[0] + 2, dtype=np.float64)
    lefts = np.empty(8, dtype=np.float64)
    rights = np.empty(8, dtype=np.float64)
    new_lefts = np.empty(8, dtype=np.float64)
    new_rights = np.empty(8, dtype=np.float64)
    edge_lefts = np.empty(2, dtype=np.float64)
    edge_rights = np.empty(2, dtype=np.float64)
    for index in range(first, last + 1):
        if index < 0 or index >= output.shape[0]:
            continue

        sample_time = start_time + index * dt
        bin_start = sample_time - 0.5 * dt - delay
        bin_end = sample_time + 0.5 * dt - delay
        if bin_end <= 0.0:
            continue

        event_count = 1
        event_times[0] = bin_start
        for radius_index in range(critical_radii.shape[0]):
            radius = critical_radii[radius_index]
            event_time = math.sqrt(
                distance_to_plane * distance_to_plane + radius * radius
            ) / sound_speed
            if bin_start + 1e-15 < event_time < bin_end - 1e-15:
                event_times[event_count] = event_time
                event_count += 1
        event_times[event_count] = bin_end
        event_count += 1

        integral = 0.0
        for event_index in range(event_count - 1):
            left = event_times[event_index]
            right = event_times[event_index + 1]
            width = right - left
            if width <= 1e-18:
                continue
            midpoint = 0.5 * (left + right)
            half_width = 0.5 * width
            for node_index in range(nodes.shape[0]):
                tau = midpoint + half_width * nodes[node_index]
                if tau <= 0.0:
                    continue
                radius = sound_speed * tau
                if radius < distance_to_plane:
                    continue
                in_plane = math.sqrt(
                    max(radius * radius - distance_to_plane * distance_to_plane, 0.0)
                )
                angle = _convex_polygon_angular_measure(
                    normal_angles,
                    thresholds,
                    origin_inside,
                    in_plane,
                    lefts,
                    rights,
                    new_lefts,
                    new_rights,
                    edge_lefts,
                    edge_rights,
                )
                if angle != 0.0:
                    value = sound_speed * angle / (2.0 * math.pi)
                    integral += weights[node_index] * half_width * value

        if integral != 0.0:
            baffle_scale = 1.0
            if has_baffle:
                propagation_time = sample_time - propagation_delay
                if propagation_time <= 0.0:
                    baffle_scale = 0.0
                else:
                    baffle_scale = baffle_distance / (sound_speed * propagation_time)
                    if baffle_scale < 0.0:
                        baffle_scale = 0.0
                output[index] += weight_scale * baffle_scale * sampling_frequency * integral


@njit(cache=True)
def prepare_flat_facets(points, facet_vertices, facet_vertex_counts, facet_normals, sound_speed):
    point_count = points.shape[0]
    facet_count = facet_vertices.shape[0]
    max_vertices = facet_vertices.shape[1]
    max_critical = 2 * max_vertices + 1

    ok = np.empty((point_count, facet_count), dtype=np.bool_)
    prepared_normals = np.empty((point_count, facet_count, 3), dtype=np.float64)
    centroids = np.empty((point_count, facet_count, 3), dtype=np.float64)
    distance_to_plane = np.empty((point_count, facet_count), dtype=np.float64)
    critical_radii = np.zeros((point_count, facet_count, max_critical), dtype=np.float64)
    critical_counts = np.zeros((point_count, facet_count), dtype=np.int64)
    normal_angles = np.zeros((point_count, facet_count, max_vertices), dtype=np.float64)
    thresholds = np.zeros((point_count, facet_count, max_vertices), dtype=np.float64)
    angular_counts = np.zeros((point_count, facet_count), dtype=np.int64)
    origin_inside = np.empty((point_count, facet_count), dtype=np.bool_)
    support_start = np.empty((point_count, facet_count), dtype=np.float64)
    support_end = np.empty((point_count, facet_count), dtype=np.float64)

    for point_index in range(point_count):
        point = points[point_index]
        for facet_index in range(facet_count):
            vertex_count = facet_vertex_counts[facet_index]
            (
                facet_ok,
                prepared_normal,
                centroid,
                distance,
                _vertices_2d,
                facet_critical_radii,
                critical_count,
                facet_normal_angles,
                facet_thresholds,
                angular_count,
                facet_origin_inside,
                facet_support_start,
                facet_support_end,
            ) = prepare_flat_facet(
                point,
                facet_vertices[facet_index, :vertex_count, :],
                facet_normals[facet_index],
                sound_speed,
            )

            ok[point_index, facet_index] = facet_ok
            origin_inside[point_index, facet_index] = facet_origin_inside
            distance_to_plane[point_index, facet_index] = distance
            support_start[point_index, facet_index] = facet_support_start
            support_end[point_index, facet_index] = facet_support_end
            for axis in range(3):
                prepared_normals[point_index, facet_index, axis] = prepared_normal[axis]
                centroids[point_index, facet_index, axis] = centroid[axis]
            if facet_ok:
                critical_counts[point_index, facet_index] = critical_count
                angular_counts[point_index, facet_index] = angular_count
                for radius_index in range(critical_count):
                    critical_radii[point_index, facet_index, radius_index] = facet_critical_radii[radius_index]
                for edge_index in range(angular_count):
                    normal_angles[point_index, facet_index, edge_index] = facet_normal_angles[edge_index]
                    thresholds[point_index, facet_index, edge_index] = facet_thresholds[edge_index]

    return (
        ok,
        prepared_normals,
        centroids,
        distance_to_plane,
        critical_radii,
        critical_counts,
        normal_angles,
        thresholds,
        angular_counts,
        origin_inside,
        support_start,
        support_end,
    )


@njit(cache=True)
def _add_exact_flat_polygon_response_column(
    output,
    point_index,
    normal_angles,
    thresholds,
    angular_count,
    origin_inside,
    critical_radii,
    critical_count,
    distance_to_plane,
    first,
    last,
    start_time,
    delay,
    propagation_delay,
    weight_scale,
    baffle_distance,
    has_baffle,
    sound_speed,
    sampling_frequency,
    nodes,
    weights,
):
    dt = 1.0 / sampling_frequency
    event_times = np.empty(critical_count + 2, dtype=np.float64)
    lefts = np.empty(8, dtype=np.float64)
    rights = np.empty(8, dtype=np.float64)
    new_lefts = np.empty(8, dtype=np.float64)
    new_rights = np.empty(8, dtype=np.float64)
    edge_lefts = np.empty(2, dtype=np.float64)
    edge_rights = np.empty(2, dtype=np.float64)
    for index in range(first, last + 1):
        if index < 0 or index >= output.shape[0]:
            continue

        sample_time = start_time + index * dt
        bin_start = sample_time - 0.5 * dt - delay
        bin_end = sample_time + 0.5 * dt - delay
        if bin_end <= 0.0:
            continue

        event_count = 1
        event_times[0] = bin_start
        for radius_index in range(critical_count):
            radius = critical_radii[radius_index]
            event_time = math.sqrt(
                distance_to_plane * distance_to_plane + radius * radius
            ) / sound_speed
            if bin_start + 1e-15 < event_time < bin_end - 1e-15:
                event_times[event_count] = event_time
                event_count += 1
        event_times[event_count] = bin_end
        event_count += 1

        integral = 0.0
        for event_index in range(event_count - 1):
            left = event_times[event_index]
            right = event_times[event_index + 1]
            width = right - left
            if width <= 1e-18:
                continue
            midpoint = 0.5 * (left + right)
            half_width = 0.5 * width
            for node_index in range(nodes.shape[0]):
                tau = midpoint + half_width * nodes[node_index]
                if tau <= 0.0:
                    continue
                radius = sound_speed * tau
                if radius < distance_to_plane:
                    continue
                in_plane = math.sqrt(
                    max(radius * radius - distance_to_plane * distance_to_plane, 0.0)
                )
                angle = _convex_polygon_angular_measure_count(
                    normal_angles,
                    thresholds,
                    angular_count,
                    origin_inside,
                    in_plane,
                    lefts,
                    rights,
                    new_lefts,
                    new_rights,
                    edge_lefts,
                    edge_rights,
                )
                if angle != 0.0:
                    value = sound_speed * angle / (2.0 * math.pi)
                    integral += weights[node_index] * half_width * value

        if integral != 0.0:
            baffle_scale = 1.0
            if has_baffle:
                propagation_time = sample_time - propagation_delay
                if propagation_time <= 0.0:
                    baffle_scale = 0.0
                else:
                    baffle_scale = baffle_distance / (sound_speed * propagation_time)
                    if baffle_scale < 0.0:
                        baffle_scale = 0.0
            output[index, point_index] += weight_scale * baffle_scale * sampling_frequency * integral


@njit(cache=True)
def add_spatial_impulse_response(
    output,
    points,
    centers,
    normals,
    areas,
    physical_indices,
    element_has_vertices,
    element_facet_start,
    element_facet_count,
    delays,
    subelement_delays,
    apodization,
    subelement_apodization,
    prepared_normals,
    prepared_centroids,
    distance_to_plane,
    critical_radii,
    critical_counts,
    normal_angles,
    thresholds,
    angular_counts,
    origin_inside,
    support_start,
    support_end,
    start_time,
    sound_speed,
    sampling_frequency,
    nodes,
    weights,
    has_soft_baffle,
):
    for point_index in range(points.shape[0]):
        point = points[point_index]
        for element_index in range(centers.shape[0]):
            physical_index = physical_indices[element_index]
            physical_delay = delays[physical_index]
            delay = physical_delay + subelement_delays[element_index]
            weight_scale = apodization[physical_index] * subelement_apodization[element_index]

            if element_has_vertices[element_index]:
                start = element_facet_start[element_index]
                end = start + element_facet_count[element_index]
                for facet_index in range(start, end):
                    facet_baffle_distance = 0.0
                    facet_has_baffle = False
                    if has_soft_baffle:
                        distance = (
                            prepared_normals[point_index, facet_index, 0]
                            * (point[0] - prepared_centroids[point_index, facet_index, 0])
                            + prepared_normals[point_index, facet_index, 1]
                            * (point[1] - prepared_centroids[point_index, facet_index, 1])
                            + prepared_normals[point_index, facet_index, 2]
                            * (point[2] - prepared_centroids[point_index, facet_index, 2])
                        )
                        if distance > 0.0:
                            facet_baffle_distance = distance
                        facet_has_baffle = True

                    first = int(
                        math.floor(
                            (support_start[point_index, facet_index] + delay - start_time)
                            * sampling_frequency
                            - 0.5
                        )
                    ) - 1
                    last = int(
                        math.ceil(
                            (support_end[point_index, facet_index] + delay - start_time)
                            * sampling_frequency
                            + 0.5
                        )
                    ) + 1
                    _add_exact_flat_polygon_response_column(
                        output,
                        point_index,
                        normal_angles[point_index, facet_index],
                        thresholds[point_index, facet_index],
                        angular_counts[point_index, facet_index],
                        origin_inside[point_index, facet_index],
                        critical_radii[point_index, facet_index],
                        critical_counts[point_index, facet_index],
                        distance_to_plane[point_index, facet_index],
                        first,
                        last,
                        start_time,
                        delay,
                        physical_delay,
                        weight_scale,
                        facet_baffle_distance,
                        facet_has_baffle,
                        sound_speed,
                        sampling_frequency,
                        nodes,
                        weights,
                    )
            else:
                dx = point[0] - centers[element_index, 0]
                dy = point[1] - centers[element_index, 1]
                dz = point[2] - centers[element_index, 2]
                geometric_distance = math.sqrt(dx * dx + dy * dy + dz * dz)
                if geometric_distance == 0.0:
                    raise ValueError("field points must not coincide with aperture elements")
                arrival_time = geometric_distance / sound_speed + delay
                sample_index = int(math.floor((arrival_time - start_time) * sampling_frequency + 0.5))
                if 0 <= sample_index < output.shape[0]:
                    sample_time = start_time + sample_index / sampling_frequency
                    propagation_time = sample_time - physical_delay
                    if propagation_time > 0.0:
                        baffle_scale = 1.0
                        if has_soft_baffle:
                            baffle_distance = normals[element_index, 0] * dx
                            baffle_distance += normals[element_index, 1] * dy
                            baffle_distance += normals[element_index, 2] * dz
                            if baffle_distance < 0.0:
                                baffle_distance = 0.0
                            distance = sound_speed * propagation_time
                            baffle_scale = baffle_distance / distance
                            if baffle_scale < 0.0:
                                baffle_scale = 0.0
                        distance = sound_speed * propagation_time
                        output[sample_index, point_index] += (
                            weight_scale
                            * baffle_scale
                            * areas[element_index]
                            / (2.0 * math.pi * distance)
                            * sampling_frequency
                        )
