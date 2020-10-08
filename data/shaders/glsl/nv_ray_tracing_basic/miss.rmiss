#version 460
#extension GL_NV_ray_tracing : require
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


layout(location = 0) rayPayloadInNV HitPy hit;

void main()
{
    hit.valid = 0;
    hit.distance = -1;
    hit.point = vec4(0);
    hit.normal = vec4(0);
    hit.bary_u = 0.0f;// draws black pixel on background
    hit.bary_v = 0.0f;
}