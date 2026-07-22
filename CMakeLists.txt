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

add_library(llmclient STATIC
    src/client.cpp
    src/types.cpp
)

target_include_directories(llmclient PUBLIC include)
target_link_libraries(llmclient PUBLIC
    CURL::libcurl
    nlohmann_json::nlohmann_json
)
