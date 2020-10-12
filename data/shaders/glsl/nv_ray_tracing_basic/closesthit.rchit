#version 460
#extension GL_NV_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
struct HitPy
{
    vec4 point;// The point in 3D space that the ray hit.
    vec4 normal;// The normalized geometry normal
    uint valid;// true if ray hit a vertex
    float distance;// The distance measured from the ray origin to this hit.
    float bary_u;// The u component of barycentric coordinate of this hit.
    float bary_v;// The v component of barycentric coordinate of this hit.
    uint instID;// The instance ID of the object in the scene
    uint primID;// The index of the primitive of the mesh hit
    uint lidar_id;// The lidar id of the ray
    uint padding;// makes structure 64bytes in size
};

layout(binding = 4, set = 0, std140) buffer Vertices { vec4 v[]; } vertices[];
layout(binding = 5, set = 0) buffer Indices { uint i[]; } indices[];

layout(location = 0) rayPayloadInNV HitPy hit;

hitAttributeNV vec2 attribs;

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
    //    hitValue = gl_WorldRayDirectionNV; // / 4.2;

    //https://nvpro-samples.github.io/vk_raytracing_tutorial/
    //const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    // Computing the normal at hit position
    //vec3 normal = v0.nrm * barycentrics.x + v1.nrm * barycentrics.y + v2.nrm * barycentrics.z;

    const ivec3 index = ivec3(indices[gl_InstanceID].i[3*gl_PrimitiveID], indices[gl_InstanceID].i[3*gl_PrimitiveID + 1], indices[gl_InstanceID].i[3*gl_PrimitiveID + 2]);
    const vec3 v0 = vertices[gl_InstanceID].v[index.x].xyz;
    const vec3 v1 = vertices[gl_InstanceID].v[index.y].xyz;
    const vec3 v2 = vertices[gl_InstanceID].v[index.z].xyz;
    const vec3 crs = cross((v1 - v0), (v2 - v0));
    const vec3 nrm = normalize(crs);
    const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    const vec3 normal = nrm * barycentrics.x + nrm * barycentrics.y + nrm * barycentrics.z;


    //    VertexObj& v0 = m_vertices[m_indices[i + 0]];
    //    VertexObj& v1 = m_vertices[m_indices[i + 1]];
    //    VertexObj& v2 = m_vertices[m_indices[i + 2]];
    //
    //    nvmath::vec3f n = nvmath::normalize(nvmath::cross((v1.pos - v0.pos), (v2.pos - v0.pos)));

    //    uint  lidar_id;// The lidar id of the ray

    hit.valid = 1;
    hit.normal = vec4(crs.xyz, 0.0f);
    hit.primID = gl_PrimitiveID;
    hit.instID = gl_InstanceCustomIndexNV;
    hit.distance = gl_RayTmaxNV;
    hit.bary_u = attribs.x;
    hit.bary_v = attribs.y;
    hit.point = vec4(gl_WorldRayOriginNV + gl_RayTmaxNV * gl_WorldRayDirectionNV, 0.0f);

}
