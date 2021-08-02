// Call with
//     [1]   halo index (integer)
//     [2-4] "upper left" corner (3 floats)
//           [the minimum coordinates]
//     [5]   extent (float)
//
// Assumes all data stored as floats!

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <memory>
#include <vector>
#include <utility>

// where to find the files
// replace: %lu   halo index 0 ... something
//          %s    [in]  "coords",
//                      "velocities"
//                [out] "offsets" -- these files store the offsets in the sorted arrays
static const char storage[] = "/scratch/gpfs/lthiele/tSZ_DeepSet_halos/rockstar/DM_%lu_%s.bin";

// how many cells we want to use
static constexpr size_t Nside = 32;


// ---------- globals --------------

// file name buffer
char buffer[512];

// the index of the current halo, to be inserted in the format string above
size_t halo_idx;

// number of particles
size_t N;

// the upper left corner
float ul_corner[3];

// the local box size
float box_size;

// store the particle indices here
// -- first entry is particle index in original order,
//    second entry is index in cells
std::vector<std::pair<size_t, size_t>> indices {};


// ---------- forward declarations ---------------

void get_N ();

void create_indices ();

void sort_indices ();

template<size_t stride>
void reorder (const char *name);

void create_offsets ();


// ----------- driver ----------------

int main (int, char **argv)
{// {{{
    halo_idx = (size_t)std::atol(argv[1]);

    for (size_t ii=0; ii != 3; ++ii)
        ul_corner[ii] = (float)std::atof(argv[2+ii]);

    box_size = (float)std::atof(argv[5]);


    get_N();

    create_indices();

    sort_indices();

    reorder<3>("coords");
    reorder<3>("velocities");

    create_offsets();

    return 0;
}// }}}


// ----------- implementation ---------------

void get_N ()
{// {{{
    std::sprintf(buffer, storage, halo_idx, "coords");
    auto f = std::fopen(buffer, "r");
    if (!f) std::abort();
    if (std::fseek(f, 0L, SEEK_END)) std::abort();
    long Nbytes = std::ftell(f);
    if (std::fclose(f)) std::abort();

    N = (size_t)(Nbytes) / (3UL * sizeof(float));

    if (N * 3UL * sizeof(float) != (size_t)Nbytes) std::abort();
}// }}}

void create_indices ()
{// {{{
    indices.reserve(N);

    std::sprintf(buffer, storage, halo_idx, "coords");
    auto f = std::fopen(buffer, "r");
    if (!f) std::abort();

    float *x = (float *)std::malloc(N * 3UL * sizeof(float));
    size_t Nread = std::fread(x, sizeof(float), 3UL * N, f);
    if (Nread != 3UL * N) std::abort();
    if (std::fclose(f)) std::abort();

    // subtract the zero point
    for (size_t ii=0; ii != N; ++ii)
        for (size_t jj=0; jj != 3; ++jj)
            x[ii*3UL+jj] -= ul_corner[jj];

    // the cell sidelength
    float acell = box_size / Nside;

    #define GRID(r, dir) (std::min((size_t)(r[dir]/acell), Nside-1UL))

    float *this_x = x;
    for (size_t ii=0; ii != N; ++ii, this_x += 3)
        indices.emplace_back(ii, Nside * Nside * GRID(this_x, 0)
                                 +       Nside * GRID(this_x, 1)
                                 +               GRID(this_x, 2));

    #undef GRID

    std::free(x);
}// }}}

void sort_indices ()
{// {{{
    std::sort(indices.begin(), indices.end(),
              [](std::pair<size_t, size_t> a,
                 std::pair<size_t, size_t> b)
              { return a.second < b.second; } );
}// }}}

template<size_t stride>
void reorder (const char *name)
{// {{{
    std::sprintf(buffer, storage, halo_idx, name);
    auto fin = std::fopen(buffer, "r");
    if (!fin) std::abort();

    float *in = (float *)std::malloc(N * stride * sizeof(float));
    float *out = (float *)std::malloc(N * stride * sizeof(float));

    size_t Nread = std::fread(in, sizeof(float), N * stride, fin);
    if (Nread != N * stride) std::abort();
    if (std::fclose(fin)) std::abort();

    assert(indices.size() == N);

    for (size_t ii=0; ii != N; ++ii)
        std::memcpy(out + ii * stride,
                    in + indices[ii].first * stride,
                    stride * sizeof(float)); 

    std::free(in);

    auto fout = std::fopen(buffer, "w");
    if (!fout) std::abort();

    size_t Nwritten = std::fwrite(out, sizeof(float), N * stride, fout);
    if (Nwritten != N * stride) std::abort();
    if (std::fclose(fout)) std::abort();

    std::free(out);
}// }}}

void create_offsets ()
{// {{{
    const size_t Ncells = Nside * Nside * Nside;

    size_t *offsets = (size_t *)std::malloc((Ncells + 1UL) * sizeof(size_t));
    
    for (size_t ii=0; ii != Ncells+1UL; ++ii)
        offsets[ii] = N;

    offsets[indices[0].second] = 0UL;

    for (size_t ii=1; ii != N; ++ii)
        if (indices[ii].second != indices[ii-1].second)
            offsets[indices[ii].second] = ii;

    std::sprintf(buffer, storage, halo_idx, "offsets");
    auto f = std::fopen(buffer, "w");
    if (!f) std::abort();

    size_t Nwritten = std::fwrite(offsets, sizeof(size_t), Ncells+1UL, f);
    if (Nwritten != Ncells+1UL) std::abort();

    if (std::fclose(f)) std::abort();

    std::free(offsets);
}// }}}
