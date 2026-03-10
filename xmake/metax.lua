-- =============================================================
-- 沐曦 (MetaX) C500 GPU 构建规则
-- =============================================================
-- 用法: xmake build --metax-gpu=true
--
-- 当 MXMACA SDK 可用时，追加 ENABLE_METAX_RUNTIME 以启用真实实现。
-- 否则仅编译 stub 骨架（getDeviceCount 返回 0，其余抛异常）。
-- =============================================================

target("llaisys-device-metax")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_defines("ENABLE_METAX_API")

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/device/metax/*.cpp")

    -- 如果 MXMACA SDK 存在，启用真实 runtime 实现
    -- 用户可通过 --metax-sdk=/opt/mxmaca 指定 SDK 路径
    -- 默认检查 /opt/mxmaca
    if os.isdir("/opt/mxmaca/include") then
        add_defines("ENABLE_METAX_RUNTIME")
        add_includedirs("/opt/mxmaca/include")
        add_linkdirs("/opt/mxmaca/lib")
        add_links("maca_runtime")
    end

    on_install(function (target) end)
target_end()

-- 沐曦算子库
target("llaisys-ops-metax")
    set_kind("static")
    set_languages("cxx17")
    add_deps("llaisys-tensor")
    set_warnings("all", "error")
    add_defines("ENABLE_METAX_API")

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    if os.isdir("/opt/mxmaca/include") then
        -- 在 MetaX 平台上编译真实 .cu 算子内核
        add_defines("ENABLE_METAX_RUNTIME")
        add_includedirs("/opt/mxmaca/include")
        add_linkdirs("/opt/mxmaca/lib")
        add_links("maca_runtime", "mxblas")
        add_files("../src/ops/*/metax/*.cu")
    else
        -- 在非 MetaX 平台上编译桩实现（提供符号定义，运行时抛异常）
        add_files("../src/ops/metax_stubs.cpp")
    end

    on_install(function (target) end)
target_end()
