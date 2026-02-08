# Tests for dimensions.jl: Grid Dimensions and Resolution Handling

using Test
# Import from the nested TransportDimensions module
using NuclearDetonation.Transport.TransportDimensions: nx, ny, nk, surface_index, output_resolution_factor,
                                                      set_dimensions!, hres_field, lres_pos, hres_pos

@testset "Grid Dimensions" begin

    @testset "Initial Grid Dimensions" begin
        # Test initial values
        @test nx[] == 864  # Default NXPRE
        @test ny[] == 698  # Default NYPRE
        @test nk[] == 61   # Default NKPRE
    end

    @testset "Setting Grid Dimensions" begin
        # Save original values
        orig_nx, orig_ny, orig_nk = nx[], ny[], nk[]
        
        try
            # Set new dimensions
            set_dimensions!(100, 100, 40)
            @test nx[] == 100
            @test ny[] == 100
            @test nk[] == 40
            
            # Test with custom surface index and output factor
            set_dimensions!(50, 50, 20, surface_idx=2, output_res_factor=2)
            @test nx[] == 50
            @test ny[] == 50
            @test nk[] == 20
            @test surface_index[] == 2
            @test output_resolution_factor[] == 2
        finally
            # Restore original values
            set_dimensions!(orig_nx, orig_ny, orig_nk)
        end
    end

    @testset "Field Upscaling" begin
        # Set dimensions for test
        set_dimensions!(2, 2, 1, output_res_factor=2)
        
        # Create a simple 2x2 field
        field_in = Float32[1.0 2.0; 3.0 4.0]

        # Upscale by 2x
        field_out = hres_field(field_in, false)  # Nearest neighbor
        @test size(field_out) == (4, 4)

        # Check that values are repeated correctly (nearest neighbor)
        @test field_out[1, 1] == 1.0
        @test field_out[2, 1] == 1.0
        @test field_out[1, 2] == 1.0
        @test field_out[2, 2] == 1.0
        
        # Check other positions
        @test field_out[3, 1] == 3.0  # From position [2,1] of original
        @test field_out[4, 1] == 3.0
        @test field_out[3, 2] == 3.0
        @test field_out[4, 2] == 3.0
    end

    @testset "Position Conversion" begin
        # Test high-resolution position to low-resolution position conversion
        output_resolution_factor[] = 2
        
        # Position in high-res grid -> low-res grid
        # With factor=2: hres_pos(1)=0, hres_pos(2)=1, hres_pos(3)=2, hres_pos(4)=2
        @test lres_pos(1) == 0  # First high-res cell maps to low-res cell 0
        @test lres_pos(2) == 1  # Second high-res cell maps to low-res cell 1
        @test lres_pos(3) == 2  # Third high-res cell maps to low-res cell 2
        @test lres_pos(4) == 2  # Fourth high-res cell still maps to low-res cell 2
        
        # Test low-resolution position to high-resolution position conversion
        # With factor=2: lres_pos(1.0)=2, lres_pos(2.0)=4
        @test hres_pos(1.0) == 2  # First low-res cell center maps to high-res pos 2
        @test hres_pos(2.0) == 4  # Second low-res cell center maps to high-res pos 4
    end

    @testset "Surface Index" begin
        # Test default value
        @test surface_index[] == -1  # Default value
        
        # Test setting it to a specific value
        set_dimensions!(100, 100, 40, surface_idx=1)
        @test surface_index[] == 1
    end

    println("✓ All Dimensions tests passed!")
end
