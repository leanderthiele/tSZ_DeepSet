// compile to shared library, exports only the prtfinder function as a symbol

#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <utility>

// -------------- constants ------------------

static const constexpr size_t Nside = 32; // cells per side
static const constexpr size_t Ninit = 128; // initial malloc size


// ------------ forward declarations ------------------

static inline size_t
periodic_idx (int idx);

static inline bool
sph_cub_intersect (const float *x0, // origin of the sphere
                   float *cub, // coordinate of the cube (ul_corner)
                   float Rsq // squared radius of the sphere
                  );

static inline void
mod_translations (const float *x0, float *cub);

static inline void
mod_reflections (float *cub);


static std::vector<std::pair<size_t, size_t>>
find_idx_ranges (const float *x0,
                 float R,
                 uint64_t Nin,
                 const float *ul_corner,
                 float extent,
                 const uint64_t *offsets,
                 int *err
                );

static uint64_t *
find_indices (const float *x0,
              float R,
              const float *x,
              uint64_t *Nout,
              const std::vector<std::pair<size_t, size_t>> *idx_ranges,
              int *err
             );


static uint64_t *
prtfinder_(const float *x0, // origin around which to find particles [3]
           float R, // radius of the sphere
           const float *x, // particle coordinates [Nin x 3]
           uint64_t Nin, // number of particles
           const float *ul_corner, // minimum coordinates [3]
           float extent, // local box size
           const uint64_t *offsets, // see sort_particles.cpp, [Nside*Nside*Nside+1]
           uint64_t *Nout, // length of output array
           int *err // error flag (nonzero if error occured)
          );


// -------------- code to be called from python's ctypes ---------------
// (we need to turn off name mangling here)
//
extern "C"
{
    // NOTE the coordinates x, as well as the ul_corner already have the halo position
    //      subtracted off, so we do not need to worry about periodic boundary conditions!
    uint64_t * prtfinder (const float *x0, // origin around which to find particles [3]
                          float R, // radius of the sphere
                          const float *x, // particle coordinates [Nin x 3]
                          uint64_t Nin, // number of particles
                          const float *ul_corner, // minimum coordinates [3]
                          float extent, // local box size
                          const uint64_t *offsets, // see sort_particles.cpp, [Nside*Nside*Nside+1]
                          uint64_t *Nout, // length of output array
                          int *err // error flag (nonzero if error occured)
                         )
    {
        return prtfinder_(x0, R, x, Nin, ul_corner, extent, offsets, Nout, err);
    }
} // extern "C"


// --------------- implementation -------------------

static uint64_t *
prtfinder_(const float *x0, // origin around which to find particles [3]
           float R, // radius of the sphere
           const float *x, // particle coordinates [Nin x 3]
           uint64_t Nin, // number of particles
           const float *ul_corner, // minimum coordinates [3]
           float extent, // local box size
           const uint64_t *offsets, // see sort_particles.cpp, [Nside*Nside*Nside+1]
           uint64_t *Nout, // length of output array
           int *err // error flag (nonzero if error occured)
          )
{// {{{
    int tmp_err;

    auto idx_ranges = find_idx_ranges(x0, R, Nin, ul_corner, extent, offsets, &tmp_err);
    if (tmp_err)
    {
        *err = 1;
        return nullptr;
    }

    auto out = find_indices(x0, R, x, Nout, &idx_ranges, &tmp_err);
    if (tmp_err)
    {
        *err = 2;
        return nullptr;
    }

    *err = 0;
    return out;
}// }}}


static std::vector<std::pair<size_t, size_t>>
find_idx_ranges (const float *x0,
                 float R,
                 uint64_t Nin,
                 const float *ul_corner,
                 float extent,
                 const uint64_t *offsets,
                 int *err
                )
{// {{{
    std::vector<std::pair<size_t, size_t>> out;

    // normalize to unit cell size and zero ul_corner
    const float acell = extent / Nside;

    float x0_normalized[3];
    for (size_t ii=0; ii != 3; ++ii)
        x0_normalized[ii] = (x0[ii]-ul_corner[ii]) / acell;

    float R_normalized = R / acell;
    float Rsq_normalized = R_normalized * R_normalized;

    // now do the looping
    for (int xx  = (int)(x0_normalized[0] - R_normalized) - 1;
             xx <= (int)(x0_normalized[0] + R_normalized);
           ++xx)
    {
        const size_t idx_x = Nside * Nside * periodic_idx(xx);

        for (int yy  = (int)(x0_normalized[1] - R_normalized) - 1;
                 yy <= (int)(x0_normalized[1] + R_normalized);
               ++yy)
        {
            const size_t idx_y = idx_x + Nside * periodic_idx(yy);

            for (int zz  = (int)(x0_normalized[2] - R_normalized) - 1;
                     zz <= (int)(x0_normalized[2] + R_normalized);
                   ++zz)
            {
                const size_t idx = idx_y + periodic_idx(zz);

                float cub[] = { (float)xx, (float)yy, (float)zz };

                if (sph_cub_intersect(x0, cub, Rsq_normalized)
                    && offsets[idx] < Nin)
                {
                    std::pair<size_t, size_t> idx_range { offsets[idx], Nin };

                    for (size_t ii=(size_t)idx+1UL; ii != Nside*Nside*Nside; ++ii)
                        if (offsets[ii] < Nin)
                        {
                            idx_range.second = offsets[ii];
                            break;
                        }

                    out.push_back(idx_range);
                }
            }
        }
    }

    *err = 0;
    return out;
}// }}}


static uint64_t *
find_indices (const float *x0,
              float Rsq,
              const float *x,
              uint64_t *Nout,
              const std::vector<std::pair<size_t, size_t>> *idx_ranges,
              int *err
             )
{// {{{
    // this case is pathological
    if (idx_ranges->empty())
    {
        *err = -1;
        return nullptr;
    }

    // initial memory allocation
    *Nout = 0UL;
    size_t Nalloc = Ninit;
    uint64_t *out = (uint64_t *)std::malloc(Nalloc * sizeof(uint64_t));

    for (const auto &idx_range : *idx_ranges)
        for (size_t ii=idx_range.first; ii != idx_range.second; ++ii)
            #define SQU(x_) (x_*x_)
            if (SQU(x[ii*3+0]-x0[0]) + SQU(x[ii*3+1]-x0[1]) + SQU(x[ii*3+2]-x0[2])
                < Rsq)
            #undef SQU
            {
                // check if we need to make more space
                if (*Nout == Nalloc)
                {
                    Nalloc *= 2;
                    out = (uint64_t *)std::realloc(out, Nalloc * sizeof(uint64_t));
                }

                out[(*Nout)++] = ii;
            }

    // shrink to correct size
    out = (uint64_t *)std::realloc(out, (*Nout) * sizeof(uint64_t));
    
    *err = 0;
    return out;
}// }}}


static inline size_t
periodic_idx (int idx)
{// {{{
    static constexpr int N = (int)Nside;
    return (N + idx%N) % N;
}// }}}


static inline bool
sph_cub_intersect (const float *x0,
                   float *cub,
                   float Rsq
                  )
{// {{{
    mod_translations(x0, cub);
    mod_reflections(cub);

    #define SQU(x_) (x_*x_)
    return SQU(std::max(0.0F, cub[0])) + SQU(std::max(0.0F, cub[1])) + SQU(std::max(0.0F, cub[2]))
           < Rsq;
    #undef SQU
}// }}}


static inline void
mod_translations (const float *x0, float *cub)
{// {{{
    for (size_t ii=0; ii != 3; ++ii)
        cub[ii] -= x0[ii];
}// }}}


static inline void
mod_reflections (float *cub)
{// {{{
    for (size_t ii=0; ii != 3; ++ii)
        if (cub[ii] < -0.5F)
            cub[ii] = - (cub[ii] + 1.0F);
}// }}}
