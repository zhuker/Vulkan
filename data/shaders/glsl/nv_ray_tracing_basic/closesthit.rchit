#version 460
#extension GL_NV_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) rayPayloadInNV vec3 hitValue;
hitAttributeNV vec2 attribs;
//hitAttributeNV vec2 baryCoord;

vec3 rgb(float value) {
    float ratio = 2 * (value) / 1;
    float b = max(0, (1 - ratio));
    float r = max(0, (ratio - 1));
    float g = 1 - b - r;
    return vec3(r, g, b);
}

void main()
{
    //    const float d = sqrt((attribs.x - attribs.y) * (attribs.x - attribs.y));
    //  const vec3 barycentricCoords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);

    // gl_RayTmaxNV - In the closest-hit shader, the value reflects
    //    the closest distance to the intersected primitive
    //    hitValue = vec3(gl_RayTmaxNV, gl_RayTmaxNV, gl_RayTmaxNV);

    // gl_WorldRayOriginNV, gl_WorldRayDirectionNV, gl_ObjectRayOriginNV
    // the origin and direction of the
    //    ray being processed in both world and object space respectively.
    hitValue = gl_WorldRayDirectionNV; // / 4.2;
}
