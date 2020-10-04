#version 460
#extension GL_NV_ray_tracing : require
struct HitPy {
    vec3  point;// The point in 3D space that the ray hit.
    uint  valid;
    vec3  normal;// The normalized geometry normal
    float distance;// The distance measured from the ray origin to this hit.
    uint  geomID;// The geometry ID of object in the scene (ignore for now)
    uint  instID;// The instance ID of the object in the scene
    uint  primID;// The index of the primitive of the mesh hit
    uint  lidar_id;// The lidar id of the ray
    vec4 padding;
};

layout(location = 0) rayPayloadInNV HitPy hit;

void main()
{
    hit.valid = 0;
    hit.distance = -1;
}