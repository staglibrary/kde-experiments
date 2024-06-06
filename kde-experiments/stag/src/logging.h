/**
 * Macros and methods for writing log messages.
 */
#ifndef KDE_EXPERIMENTS_LOGGING_H
#define KDE_EXPERIMENTS_LOGGING_H

#ifndef NDEBUG
#  define LOG_DEBUG(x) do { std::cerr << x; } while (0)
#else
#  define LOG_DEBUG(x)
#endif

#define LOG_INFO(x) do { std::cout << x; } while (0)

#define LOG_ERROR(x) do { std::cerr << x; } while (0)

#endif //KDE_EXPERIMENTS_LOGGING_H
