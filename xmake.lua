add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

option("dist-nccl")
    set_default(false)
    set_showmenu(true)
    set_description("Enable NCCL distributed communication backend (stage A skeleton)")
option_end()

option("dist-mpi")
    set_default(false)
    set_showmenu(true)
    set_description("Enable MPI distributed communication backend (stage A skeleton)")
option_end()

-- MetaX (沐曦) --
option("metax-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for MetaX C500 GPU (MXMACA)")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end

if has_config("metax-gpu") then
    add_defines("ENABLE_METAX_API")
    includes("xmake/metax.lua")
end

if has_config("dist-nccl") then
    add_defines("ENABLE_DIST_NCCL")
end

if has_config("dist-mpi") then
    add_defines("ENABLE_DIST_MPI")
end

target("llaisys-utils")
    set_kind("static")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/utils/*.cpp")

    on_install(function (target) end)
target_end()


target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")
    if has_config("metax-gpu") then
        add_deps("llaisys-device-metax")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/device/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/core/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/tensor/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("src/ops/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")
    if has_config("metax-gpu") then
        add_deps("llaisys-ops-metax")
    end
    if has_config("nv-gpu") then
        add_links("cublas", "cudart")
        add_linkdirs("/usr/local/cuda/lib64")
        set_toolset("cu", "nvcc")
        add_cuflags("-Xcompiler=-fPIC")
        add_files("src/device/nvidia/*.cu")
        add_files("src/ops/*/nvidia/*.cu")
    end

    if has_config("metax-gpu") then
        -- 在 MetaX 平台上链接 MXMACA 运行时和 mxBLAS
        if os.isdir("/opt/mxmaca/include") then
            add_includedirs("/opt/mxmaca/include")
            add_linkdirs("/opt/mxmaca/lib")
            add_links("maca_runtime", "mxblas")
            -- mxcc 编译器兼容 CUDA 语法，编译 .cu 文件
            add_files("src/device/metax/*.cu|*.cpp")
            add_files("src/ops/*/metax/*.cu")
        end
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    
    -- 原有的接口文件
    add_files("src/llaisys/*.cc")
    
    -- 【关键修复】添加这一行以编译 Qwen2 模型实现
    add_files("src/llaisys/models/*.cpp")
    add_files("src/distributed/*.cpp")

    set_installdir(".")

    
    after_install(function (target)
        -- copy shared library to python package
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end
    end)
target_end()

target("llaisys-dist-smoke")
    set_kind("binary")
    add_deps("llaisys")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("test/dist_smoke.cpp")
target_end()

target("llaisys-tp-shard-smoke")
    set_kind("binary")
    add_deps("llaisys")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("test/tp_shard_smoke.cpp")
    add_includedirs("$(projectdir)")
target_end()

target("llaisys-tp-fwd-smoke")
    set_kind("binary")
    add_deps("llaisys")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("test/tp_fwd_smoke.cpp")
target_end()

target("llaisys-tp-cache-smoke")
    set_kind("binary")
    add_deps("llaisys")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("test/tp_cache_smoke.cpp")
target_end()