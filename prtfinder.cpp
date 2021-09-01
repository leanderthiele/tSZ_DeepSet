// compile to shared library, exports only the prtfinder function as a symbol
//
// $ g++ --std=c++17 --shared -fPIC -Wall -Wextra -O3 -o libprtfinder.so prtfinder.cpp

#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

static inline float *
load_x_file (std::FILE *x_file,
             size_t offset,
             size_t N,
             float box_size,
             const float *halo_pos,
             int *err
            );

static std::vector<std::pair<size_t, size_t>>
find_idx_ranges (const float *x0,
                 float R,
                 uint64_t Nin,
                 float R200c,
                 const uint64_t *offsets,
                 int *err
                );

static uint64_t *
find_indices (const float *x0,
              float R,
              int x_filled,
              const float *x,
              const char *x_fname,
              float box_size,
              const float *halo_pos,
              uint64_t *Nout,
              const std::vector<std::pair<size_t, size_t>> *idx_ranges,
              int *err
             );


static uint64_t *
prtfinder_(const float *x0, // origin around which to find particles [3]
           float R, // radius of the sphere
           int x_filled, // whether x is already allocated and filled
           const float *x, // particle coordinates [Nin x 3]
           const char *x_fname, // file with particle coordinates
           uint64_t Nin, // number of particles
           float box_size, // global box size
           float R200c, // halo radius
           const float *halo_pos, // halo position
           const uint64_t *offsets, // see sort_particles.cpp, [Nside*Nside*Nside+1]
           uint64_t *Nout, // length of output array
           int *err // error flag (nonzero if error occured)
          );


// -------------- code to be called from python's ctypes ---------------
// (we need to turn off name mangling here)
//
extern "C"
{
    // NOTE the coordinates x, x0 already have the halo position
    //      subtracted off, so we do not need to worry about periodic boundary conditions!
    // NOTE the argument x is only used if x_filled = True
    // NOTE if x_fname != 'NONE', x is not used but box_size, halo_pos used!
    uint64_t * prtfinder (const float *x0, // origin around which to find particles [3]
                          float R, // radius of the sphere
                          int x_filled, // whether x is already allocated and filled
                          const float *x, // particle coordinates [Nin x 3]
                          const char *x_fname, // file with particle coordinates
                          uint64_t Nin, // number of particles
                          float box_size, // global box size
                          float R200c, // halo radius
                          const float *halo_pos, // halo position
                          const uint64_t *offsets, // see sort_particles.cpp, [Nside*Nside*Nside+1]
                          uint64_t *Nout, // length of output array
                          int *err // error flag (nonzero if error occured)
                         )
    {
        return prtfinder_(x0, R, x_filled, x, x_fname, Nin, box_size, R200c, halo_pos, offsets, Nout, err);
    }

    void myfree (uint64_t *data)
    {
        std::free(data);
    }
} // extern "C"


// --------------- implementation -------------------

static uint64_t *
prtfinder_(const float *x0, // origin around which to find particles [3]
           float R, // radius of the sphere
           int x_filled, // whether x is already allocated and filled
           const float *x, // particle coordinates [Nin x 3]
           const char *x_fname, // where to find the particle coordinates
           uint64_t Nin, // number of particles
           float box_size, // global box size
           float R200c, // radius of the halo
           const float *halo_pos, // position of the halo
           const uint64_t *offsets, // see sort_particles.cpp, [Nside*Nside*Nside+1]
           uint64_t *Nout, // length of output array
           int *err // error flag (nonzero if error occured)
          )
{// {{{
    int tmp_err;

    auto idx_ranges = find_idx_ranges(x0, R, Nin, R200c, offsets, &tmp_err);
    if (tmp_err)
    {
        *err = 1000 + tmp_err;
        return nullptr;
    }

    auto out = find_indices(x0, R, x_filled, x, x_fname, box_size, halo_pos, Nout, &idx_ranges, &tmp_err);
    if (tmp_err)
    {
        *err = 2000 + tmp_err;
        return nullptr;
    }

    *err = 0;
    return out;
}// }}}


static std::vector<std::pair<size_t, size_t>>
find_idx_ranges (const float *x0,
                 float R,
                 uint64_t Nin,
                 float R200c,
                 const uint64_t *offsets,
                 int *err
                )
{// {{{
    std::vector<std::pair<size_t, size_t>> out;

    // normalize to unit cell size and zero upper left corner
    const float acell = 2.0F * 2.51F * R200c / Nside;

    float x0_normalized[3];
    for (size_t ii=0; ii != 3; ++ii)
        x0_normalized[ii] = (x0[ii]+2.51F*R200c) / acell;

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

                if (sph_cub_intersect(x0_normalized, cub, Rsq_normalized)
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
              float R,
              int x_filled,
              const float *x,
              const char *x_fname,
              float box_size,
              const float *halo_pos,
              uint64_t *Nout,
              const std::vector<std::pair<size_t, size_t>> *idx_ranges,
              int *err
             )
{// {{{
    if (idx_ranges->empty())
    {
        *err = 0;
        *Nout = 0UL;
        return nullptr;
    }

    // get a file descriptor if necessary
    std::FILE *x_file;
    if (!x_filled)
    {
        x_file = std::fopen(x_fname, "r");
        if (!x_file)
        {
            *err = 13;
            return nullptr;
        }
    }

    // initial memory allocation
    *Nout = 0UL;
    size_t Nalloc = Ninit;
    uint64_t *out = (uint64_t *)std::malloc(Nalloc * sizeof(uint64_t));

    float Rsq = R * R;

    for (const auto &idx_range : *idx_ranges)
    {
        // number of particles in this index range
        size_t Nprt = idx_range.second - idx_range.first;

        if (!Nprt)
            continue;

        const float *x_buffer = (x_filled) ?
                                // x already filled, simply set buffer to correct memory location
                                x + idx_range.first * 3UL
                                // x not filled, need to read a segment from file
                                : load_x_file(x_file, idx_range.first, Nprt, box_size, halo_pos, err);

        // if we read from file, check whether we did that successfully
        if (!x_filled && *err)
            return nullptr;

        for (size_t ii=0; ii != Nprt; ++ii)
            #define SQU(var) ((var)*(var))
            if (SQU(x_buffer[ii*3+0]-x0[0]) + SQU(x_buffer[ii*3+1]-x0[1]) + SQU(x_buffer[ii*3+2]-x0[2]) < Rsq)
            #undef SQU
            {
                // check if we need to make more space
                if (*Nout == Nalloc)
                {
                    Nalloc *= 2;
                    out = (uint64_t *)std::realloc(out, Nalloc * sizeof(uint64_t));
                }

                out[(*Nout)++] = ii + idx_range.first;
            }
        
        if (!x_filled)
            // need casting here because signature of free is stupid
            std::free((float *)x_buffer);
    }

    if (!x_filled)
        if (std::fclose(x_file))
        {
            *err = 16;
            return nullptr;
        }

    // shrink to correct size -- make sure we don't realloc to zero as subsequent frees
    //                           may be buggy
    out = (uint64_t *)std::realloc(out, std::max(1UL, *Nout) * sizeof(uint64_t));
    
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

    #define SQU(var) ((var)*(var))
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

static inline float *
load_x_file (std::FILE *x_file,
             size_t offset,
             size_t N,
             float box_size,
             const float *halo_pos,
             int *err
            )
{// {{{
    float *x_buffer = (float *)std::malloc(N * 3UL * sizeof(float));

    // in this case, we first need to read the relevant portion from disk
    if (std::fseek(x_file, (long)(offset * 3UL * sizeof(float)), SEEK_SET))
    {
        *err = 14;
        return nullptr;
    }

    size_t Nread = std::fread(x_buffer, sizeof(float), 3UL * N, x_file);
    if (Nread != 3UL * N)
    {
        *err = 15;
        return nullptr;
    }

    // now normalize the coordinates with respect to halo position and radius
    for (size_t ii=0; ii != N; ++ii)
        for (size_t dd=0; dd != 3; ++dd)
        {
            // get pointer to the element we'll operate on
            float *this_x = x_buffer + ii*3 + dd;

            // take relative to halo position
            *this_x -= halo_pos[dd];

            // enforce periodic boundary conditions
            if (*this_x > +0.5*box_size)
                *this_x -= box_size;
            else if (*this_x < -0.5*box_size)
                *this_x += box_size;
        }

    *err = 0;
    return x_buffer;
}// }}}

