#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

constexpr std::uint64_t max_discover_mem = 1 << 15;
constexpr std::uint64_t max_check_mem = 1 << 24;
constexpr std::uint64_t page_size = 16384;

using measure_t = std::chrono::duration<double>;

struct
{
    bool enabled = false;

    template <typename T>
    auto& operator<<(T&& value)
    {
        if (enabled)
        {
            std::cerr << std::forward<T>(value);
        }
        return *this;
    }
} debug_logger;

struct aligned_array
{
    std::size_t size;
    std::uint64_t* begin;

    aligned_array(std::size_t size) : size(size)
    {
        std::uint64_t bytes = size * sizeof(std::uint64_t);
        if (bytes % page_size)
        {
            bytes = (page_size / page_size + 1) * page_size;
        }
        begin = static_cast<std::uint64_t*>(std::aligned_alloc(page_size, bytes));
    }

    std::uint64_t& operator[](std::size_t i)
    {
        return begin[i];
    }

    ~aligned_array()
    {
        std::free(begin);
    }
};

volatile int sink;

aligned_array clutter(1 << 23);
aligned_array a(max_check_mem);

std::uint64_t rng_seed = 0;

std::uint64_t rng()
{
    rng_seed *= 2862933555777941757ull;
    rng_seed += 3037000493ull;
    return rng_seed;
}

void shuffle(std::uint64_t stride, std::uint64_t spots)
{
    for (std::uint32_t i = spots - 1; i > 0; --i)
    {
        std::swap(a[i * stride], a[(rng() % (i + 1)) * stride]);
    }
}

void create_forward_chain(std::uint64_t stride, std::uint64_t spots)
{
    std::uint64_t size = stride * spots;

    for (std::uint64_t i = 0; i < spots - 1; i += 1)
    {
        std::uint64_t* next = &a[(i + 1) * stride];
        a[i * stride] = reinterpret_cast<std::uint64_t>(next);
    }
    a[size - stride] = reinterpret_cast<std::uint64_t>(&a[0]);
}

auto measure_walk(std::uint64_t stride, std::uint64_t spots, std::uint64_t initial_offset)
{
    constexpr std::uint64_t ops = 1 << 20;

    std::uint64_t sum = 0;

    for (std::uint32_t i = 0; i < clutter.size; ++i)
    {
        sum += clutter[i];
        clutter[i] ^= 1;
    }

    std::uint64_t* ptr = &a[0];
    for (std::uint64_t i = 0; i < ops; ++i)
    {
        sum += *ptr;
        ptr = reinterpret_cast<std::uint64_t*>(*ptr);
    }

    ptr = &a[initial_offset];
    auto start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < ops; ++i)
    {
        sum += *ptr;
        ptr = reinterpret_cast<std::uint64_t*>(*ptr);
    }
    auto end = std::chrono::steady_clock::now();

    sink += sum;

    return (end - start);
}

auto measure_shuffled(std::uint64_t stride, std::uint64_t spots)
{
    create_forward_chain(stride, spots);
    shuffle(stride, spots);

    for (std::uint64_t i = 0; i < spots; ++i)
    {
        a[i * stride + stride / 2] = a[i * stride] + (stride / 2) * sizeof(std::uint64_t);
    }

    return measure_walk(stride, spots, stride / 2);
}

auto measure_lookbehind(std::uint64_t stride, std::uint64_t spots)
{
    constexpr std::uint32_t lookbehind_offset = 16;

    create_forward_chain(stride, spots);

    for (std::uint32_t i = 0; i < spots; ++i)
    {
        std::uint64_t behind_index = (i + spots - lookbehind_offset) % spots;
        std::uint64_t* behind_ptr = &a[behind_index * stride + stride / 2];
        *behind_ptr = a[i * stride];
        a[i * stride] = reinterpret_cast<std::uint64_t>(behind_ptr);
    }

    return measure_walk(stride, spots, 0);
}

std::uint64_t next_spots(std::uint64_t stride, std::uint64_t spots)
{
    constexpr std::uint64_t step = 1 << 9;

    if (spots * stride <= step)
    {
        return spots * 2;
    }
    else if (stride > step)
    {
        return spots + 1;
    }
    else
    {
        return spots + step / stride;
    }
}

template <typename Container>
auto median(Container& container)
{
    std::sort(container.begin(), container.end());
    return container[container.size() / 2];
}

template <typename Container>
auto sum(Container const& container)
{
    auto result = container[0];
    for (std::size_t i = 1; i < container.size(); ++i)
    {
        result += container[i];
    }
    return result;
}

template <typename Spots, typename Method>
std::map<std::uint64_t, std::vector<measure_t>>
measure(std::uint64_t stride, Spots const& all_spots, Method const& method)
{
    constexpr int iterations = 9;

    std::map<std::uint64_t, std::vector<measure_t>> results;

    for (int i = 0; i < iterations; ++i)
    {
        std::cerr << "stride " << stride << "; iteration " << i + 1 << " out of " << iterations
                  << "\r";
        debug_logger << "\n";
        for (std::uint64_t spots : all_spots)
        {
            rng_seed = spots + i;
            results[spots].push_back(method(stride, spots));
        }
    }

    return results;
}

std::vector<std::uint64_t> make_spots(std::uint64_t stride)
{
    std::vector<std::uint64_t> result;
    std::uint64_t max_spots = next_spots(stride, max_discover_mem / stride);
    for (std::uint64_t spots = 1; spots <= max_spots; spots = next_spots(stride, spots))
    {
        result.push_back(spots);
    }
    return result;
}

std::vector<std::uint64_t> find_rest_spots(std::uint64_t initial_stride)
{
    std::uint64_t stride = initial_stride;

    std::vector<std::uint64_t> all_spots = make_spots(stride);
    auto prev_measured = measure(stride, all_spots, measure_shuffled);

    while (true)
    {
        stride *= 2;
        std::set<std::uint64_t> new_spots;
        for (auto const& spots : all_spots)
        {
            new_spots.insert(spots);

            if (spots > 1)
            {
                new_spots.insert(spots / 2);
            }
        }
        auto new_measured = measure(stride, new_spots, measure_shuffled);

        double full_distance = 0;
        double half_distance = 0;
        for (std::size_t i = 0; i < all_spots.size(); ++i)
        {
            auto spots = all_spots[i];
            double diff = sum(prev_measured[spots]).count();
            diff -= sum(new_measured[spots]).count();
            full_distance += std::cbrt(std::abs(diff));
            diff = sum(prev_measured[spots]).count();
            diff -= sum(new_measured[spots / 2 + spots % 2]).count();
            half_distance += std::cbrt(std::abs(diff));
        }
        debug_logger << "full_distance=" << full_distance << " half_distance=" << half_distance
                     << "\n";
        if (full_distance < half_distance)
        {
            debug_logger << "final stride=" << stride << "\n";
            for (auto& spots : all_spots)
            {
                debug_logger << "spots=" << spots << " prev=" << sum(prev_measured[spots]).count()
                             << " full=" << sum(new_measured[spots]).count()
                             << " half=" << sum(new_measured[spots / 2 + spots % 2]).count()
                             << "\n";
                spots *= stride / 2 / initial_stride;
            }

            return all_spots;
        }

        all_spots = make_spots(stride);
        prev_measured = new_measured;
    }
}

std::uint64_t
find_jump(std::uint64_t initial_stride, std::vector<std::uint64_t> const& search_spots)
{
    auto measured = measure(initial_stride, search_spots, measure_shuffled);
    std::vector<measure_t> smooth_suffix;

    for (auto const& spots : search_spots)
    {
        smooth_suffix.push_back(sum(measured[spots]));
    }

    for (std::size_t i = smooth_suffix.size() - 1; i > 0; --i)
    {
        constexpr double alpha = 0.5;
        smooth_suffix[i - 1] = smooth_suffix[i - 1] * alpha + smooth_suffix[i] * (1 - alpha);
    }

    for (std::size_t i = 0; i < search_spots.size(); ++i)
    {
        std::uint64_t spots = search_spots[i];
        auto result = sum(measured[spots]);
        debug_logger << "spots=" << spots << " result=" << result.count()
                     << " smooth_suffix=" << smooth_suffix[i].count() << "\n";
    }

    for (std::size_t i = 0; i < search_spots.size() - 1; ++i)
    {
        std::uint64_t spots = search_spots[i];
        auto result = sum(measured[spots]);
        auto next_result = sum(measured[search_spots[i + 1]]);

        if (next_result / result >= 1.045 && smooth_suffix[i + 1] / result >= 1.12)
        {
            debug_logger << "returning spots=" << spots << "\n";
            return spots;
        }
    }

    return 0;
}

std::uint64_t find_cache_line_size()
{
    constexpr std::uint64_t min_stride = 2;
    constexpr std::uint64_t max_stride = 128;

    std::vector<measure_t> results;

    for (std::uint64_t stride = min_stride; stride <= max_stride; stride *= 2)
    {
        std::uint64_t spots = max_check_mem / stride;
        auto measured = measure(stride, std::vector{spots}, measure_lookbehind);
        measure_t result = median(measured[spots]);
        results.push_back(result);
        debug_logger << "stride=" << stride << " result=" << result.count() << "\n";
    }

    std::uint64_t cache_line_stride = 1;
    double max_speedup = 1.0;

    for (std::size_t i = 1; i < results.size(); ++i)
    {
        std::uint64_t stride = (min_stride << i);
        measure_t result = results[i];
        measure_t previous_result = results[i - 1];
        if (result / previous_result >= max_speedup)
        {
            max_speedup = result / previous_result;
            cache_line_stride = stride / 2;
        }
    }

    return cache_line_stride;
}

int main(int argc, char** argv)
{
    constexpr std::uint64_t initial_stride = 2;

    debug_logger.enabled = argc > 1 && std::string(argv[1]) == "--verbose";

    std::cerr << "Measuring cache associativity and size\n";

    auto rest_spots = find_rest_spots(initial_stride);
    std::uint64_t jump = find_jump(initial_stride, rest_spots);

    std::cerr << "Measuring cache line size\n";

    std::uint64_t cache_line_size = find_cache_line_size();

    std::cout << "associativity=" << jump / rest_spots.front();
    std::cout << " cache_size=" << jump * initial_stride * sizeof(std::uint64_t);
    std::cout << " cache_line_size=" << cache_line_size * sizeof(std::uint64_t);
    std::cout << "\n";
}
