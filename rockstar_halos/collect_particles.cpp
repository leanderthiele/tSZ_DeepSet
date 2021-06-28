#include <cstdio>
#include <limits>

#include "group_particles.hpp"
#include "common_fields.hpp"

namespace collect
{
    constexpr const uint8_t PartType =
        #if defined DM
        1;
        #elif defined TNG
        0;
        #else
        #   error "One of DM, TNG must be defined."
        #endif

    using GrpF = GrpFields<RockstarFields::pos,
                           RockstarFields::ang_mom,
                           RockstarFields::vel,
                           RockstarFields::M200c,
                           RockstarFields::R200c,
                           RockstarFields::Xoff,
                           RockstarFields::Voff,
                           RockstarFields::Vmax,
                           RockstarFields::Vrms,
                           RockstarFields::Rs,
                           RockstarFields::rs_klypin,
                           RockstarFields::M200c_all,
                           RockstarFields::Mvir,
                           RockstarFields::M200b,
                           RockstarFields::M500c,
                           RockstarFields::M2500c,
                           RockstarFields::Spin,
                           RockstarFields::spin_bullock,
                           RockstarFields::b_to_a,
                           RockstarFields::c_to_a>;

    using PrtF = PrtFields<IllustrisFields::Coordinates,
                           IllustrisFields::Potential,
                           #if defined TNG
                           IllustrisFields::Density,
                           IllustrisFields::Masses,
                           IllustrisFields::InternalEnergy,
                           IllustrisFields::ElectronAbundance,
                           IllustrisFields::StarFormationRate
                           #elif defined DM
                           IllustrisFields::Velocities
                           #endif
                           >;

    using AF = AllFields<GrpF, PrtF>;

    class ParticleCollection
    {// {{{
        using GrpProperties = typename Callback<AF>::GrpProperties;
        using PrtProperties = typename Callback<AF>::PrtProperties;

        using value_type = float;

        std::vector<value_type> coords;
        #if defined TNG
        std::vector<value_type> masses;
        std::vector<value_type> Pth;
        #elif defined DM
        std::vector<value_type> velocities;
        #endif

        value_type M200c, R200c, Xoff, Voff, Vmax, Vrms, Rs, rs_klypin,
                   M200c_all, Mvir, M200b, M500c, M2500c,
                   Spin, spin_bullock, b_to_a, c_to_a;
        value_type pos[3];
        value_type ang_mom[3];
        value_type vel[3];

        value_type min_pot_pos[3];
        value_type min_potential_energy = std::numeric_limits<value_type>::max();

    public :
        ParticleCollection() = delete;

        ParticleCollection(const GrpProperties &grp)
        {
            M200c = grp.get<RockstarFields::M200c>();
            R200c = grp.get<RockstarFields::R200c>();
            Xoff  = grp.get<RockstarFields::Xoff>();
            Voff  = grp.get<RockstarFields::Voff>();
            Vmax  = grp.get<RockstarFields::Vmax>();
            Vrms  = grp.get<RockstarFields::Vrms>();
            Rs    = grp.get<RockstarFields::Rs>();
            rs_klypin = grp.get<RockstarFields::rs_klypin>();
            M200c_all = grp.get<RockstarFields::M200c_all>();
            Mvir = grp.get<RockstarFields::Mvir>();
            M200b = grp.get<RockstarFields::M200b>();
            M500c = grp.get<RockstarFields::M500c>();
            M2500c = grp.get<RockstarFields::M2500c>();
            Spin = grp.get<RockstarFields::Spin>();
            spin_bullock = grp.get<RockstarFields::spin_bullock>();
            b_to_a = grp.get<RockstarFields::b_to_a>();
            c_to_a = grp.get<RockstarFields::c_to_a>();

            auto r = grp.get<RockstarFields::pos>();
            for (size_t ii=0; ii != 3; ++ii)
                pos[ii] = r[ii];

            auto v = grp.get<RockstarFields::vel>();
            for (size_t ii=0; ii != 3; ++ii)
                vel[ii] = v[ii];

            auto J = grp.get<RockstarFields::ang_mom>();
            for (size_t ii=0; ii != 3; ++ii)
                ang_mom[ii] = J[ii];
        }

        void prt_insert (size_t, const GrpProperties &, const PrtProperties &prt, coord_t)
        {
            auto r = prt.get<IllustrisFields::Coordinates>();
            // do not use it yet, since for TNG we want to filter for SFR

            #if defined TNG
            // we only include non-star-forming particles
            auto SFR = prt.get<IllustrisFields::StarFormationRate>();
            if (SFR > 0.0)
                return;

            auto m = prt.get<IllustrisFields::Masses>();
            masses.push_back(m);

            auto d = prt.get<IllustrisFields::Density>();
            auto x = prt.get<IllustrisFields::ElectronAbundance>();
            auto e = prt.get<IllustrisFields::InternalEnergy>();

            static constexpr const value_type gamma = 5.0/3.0, XH = 0.76;
            Pth.push_back(4.0 * x * XH / (1.0+3.0*XH+4.0*XH*x)
                          * (gamma-1.0) * d * e);
            #elif defined DM
            auto v = prt.get<IllustrisFields::Velocities>();
            for (size_t ii=0; ii != 3; ++ii)
                velocities.push_back(v[ii]);
            #endif

            auto phi = prt.get<IllustrisFields::Potential>();
            if (phi < min_potential_energy)
            {
                min_potential_energy = phi;
                for (size_t ii=0; ii != 3; ++ii)
                    min_pot_pos[ii] = r[ii];
            }

            for (size_t ii=0; ii != 3; ++ii)
                coords.push_back(r[ii]);
        }

        void save(std::FILE *fglobals, std::FILE *fcoords,
                  #if defined TNG
                  std::FILE *fmasses,
                  std::FILE *fPth
                  #elif defined DM
                  std::FILE *fvelocities
                  #endif
                  ) const
        {
            std::fprintf(fglobals, "{'M200c': %.8e, 'R200c': %.8e, 'Xoff': %.8e, 'Voff': %.8e, "
                                   " 'Vmax': %.8e, 'Vrms': %.8e, 'Rs': %.8e, 'rs_klypin': %.8e, "
                                   " 'M200c_all': %.8e, 'Mvir': %.8e, 'M200b': %.8e, 'M500c': %.8e, "
                                   " 'M2500c': %.8e, 'Spin': %.8e, 'spin_bullock': %.8e, "
                                   " 'b_to_a': %.8e, 'c_to_a': %.8e, "
                                   " 'pos': np.array([%.8e, %.8e, %.8e], dtype=np.float32), "
                                   #if defined DM
                                   " 'min_pot_pos_DM': np.array([%.8e, %.8e, %.8e], dtype=np.float32), "
                                   #elif defined TNG
                                   " 'min_pot_pos_TNG': np.array([%.8e, %.8e, %.8e], dtype=np.float32), "
                                   #endif
                                   " 'vel': np.array([%.8e, %.8e, %.8e], dtype=np.float32), "
                                   " 'ang_mom': np.array([%.8e, %.8e, %.8e], dtype=np.float32) }",
                                   M200c, R200c, Xoff, Voff,
                                   Vmax, Vrms, Rs, rs_klypin,
                                   M200c_all, Mvir, M200b, M500c,
                                   M2500c, Spin, spin_bullock,
                                   b_to_a, c_to_a,
                                   pos[0], pos[1], pos[2],
                                   min_pot_pos[0], min_pot_pos[1], min_pot_pos[2],
                                   vel[0], vel[1], vel[2],
                                   ang_mom[0], ang_mom[1], ang_mom[2]);

            std::fwrite(coords.data(), sizeof(value_type), coords.size(), fcoords);

            #if defined TNG
            std::fwrite(masses.data(), sizeof(value_type), masses.size(), fmasses);
            std::fwrite(Pth.data(), sizeof(value_type), Pth.size(), fPth);
            #elif defined DM
            std::fwrite(velocities.data(), sizeof(value_type), velocities.size(), fvelocities);
            #endif
        }
    };// }}}

    constexpr RockstarFields::R200c::value_type Rscale = 2.5; // collect all particles within 2.5 R200c
    constexpr RockstarFields::M200c::value_type Mmin = 5e3; // 5 x 10^13 Msun/h

    using grp_chunk = CallbackUtils::chunk::SingleGrp<AF>;
    using prt_chunk = CallbackUtils::chunk::MultiPrt<AF>;

    using name = CallbackUtils::name::Illustris<AF, PartType>;

    using meta = CallbackUtils::meta::Illustris<AF, PartType>;

    using grp_select_M = CallbackUtils::select::LowCutoff<AF, RockstarFields::M200c>;

    using grp_radius = CallbackUtils::radius::Simple<AF, RockstarFields::R200c>;

    using prt_collect = CallbackUtils::prt_action::StorePrtHomogeneous<AF, ParticleCollection>;
} // namespace collect

struct collect_callback :
    virtual public Callback<collect::AF>,
    public collect::grp_chunk, public collect::prt_chunk,
    public collect::name, public collect::meta,
    public collect::grp_select_M,
    public collect::grp_radius,
    public collect::prt_collect
{// {{{
    collect_callback () :
        collect::grp_chunk { fgrp }, collect::prt_chunk { fprt, prt_max_idx },
        collect::grp_select_M { collect::Mmin },
        collect::grp_radius { collect::Rscale},
        collect::prt_collect { prt_collect_v }
    { }

    std::vector<collect::ParticleCollection> prt_collect_v;

private :
    
    static constexpr const char fgrp[] = "/tigress/lthiele/Illustris_300-1_Dark/rockstar/out_99.hdf5";
    static constexpr const char fprt[] =
        #if defined DM
        "/tigress/lthiele/Illustris_300-1_Dark/output/snapdir_099/snap_099.%d.hdf5";
        #elif defined TNG
        "/tigress/lthiele/Illustris_300-1_TNG/output/snapdir_099/snap_099.%d.hdf5";
        #endif
    static constexpr const size_t prt_max_idx =
        #if defined DM
        74;
        #elif defined TNG
        599;
        #endif
};// }}}


int main ()
{
    collect_callback c;

    group_particles<> (c);

    size_t grp_idx = 0;
    char buffer[512];

    const char out_root[] =
        #if defined DM
        "/scratch/gpfs/lthiele/tSZ_DeepSet_halos/rockstar/DM";
        #elif defined TNG
        "/scratch/gpfs/lthiele/tSZ_DeepSet_halos/rockstar/TNG";
        #endif


    for (const auto &obj : c.prt_collect_v)
    {
        std::sprintf(buffer, "%s_%lu_globals.bin", out_root, grp_idx);
        auto fglobals = std::fopen(buffer, "w");

        std::sprintf(buffer, "%s_%lu_coords.bin", out_root, grp_idx);
        auto fcoords = std::fopen(buffer, "wb");

        #if defined TNG
        std::sprintf(buffer, "%s_%lu_masses.bin", out_root, grp_idx);
        auto fmasses = std::fopen(buffer, "wb");

        std::sprintf(buffer, "%s_%lu_Pth.bin", out_root, grp_idx);
        auto fPth = std::fopen(buffer, "wb");
        #elif defined DM
        std::sprintf(buffer, "%s_%lu_velocities.bin", out_root, grp_idx);
        auto fvelocities = std::fopen(buffer, "wb");
        #endif

        obj.save(fglobals, fcoords,
                 #if defined TNG
                 fmasses,
                 fPth
                 #elif defined DM
                 fvelocities
                 #endif
                 );

        ++grp_idx;
    }

    return 0;
}
