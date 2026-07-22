cmake_minimum_required(VERSION 3.25)

# Allow standalone or subdirectory use
if(NOT PROJECT_NAME)
    project(llmclient LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies with idempotent guards
find_package(CURL REQUIRED)

if(NOT TARGET nlohmann_json::nlohmann_json)
    include(FetchContent)
    FetchContent_Declare(nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# Catch2 for testing (idempotent)
if(NOT TARGET Catch2::Catch2WithMain)
    find_package(Catch2 QUIET)
    if(NOT Catch2_FOUND)
        include(FetchContent)
        FetchContent_Declare(Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG v3.7.1
        )
        FetchContent_MakeAvailable(Catch2)
    endif()
endif()

add_library(llmclient STATIC
    src/client.cpp
    src/types.cpp
)

target_include_directories(llmclient PUBLIC include)
target_link_libraries(llmclient PUBLIC
    CURL::libcurl
    nlohmann_json::nlohmann_json
)

# ── Tests ──
option(BUILD_TESTING "Build llmclient tests" ON)
if(BUILD_TESTING)
    include(CTest)
    include(Catch)

    add_executable(test_llmclient_types
        tests/test_types.cpp
    )
    target_link_libraries(test_llmclient_types PRIVATE
        llmclient
        Catch2::Catch2WithMain
    )
    target_include_directories(test_llmclient_types PRIVATE include)

    add_executable(test_llmclient_client
        tests/test_client.cpp
    )
    target_link_libraries(test_llmclient_client PRIVATE
        llmclient
        Catch2::Catch2WithMain
    )
    target_include_directories(test_llmclient_client PRIVATE include)

    catch_discover_tests(test_llmclient_types)
    catch_discover_tests(test_llmclient_client)
endif()
