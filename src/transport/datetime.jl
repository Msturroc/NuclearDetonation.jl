# SNAP: Severe Nuclear Accident Programme
# Julia port of datetime.f90
# Original Copyright (C) 1992-2023 Norwegian Meteorological Institute
#
# Represents datetime with hourly granularity
# All times in UTC (Zulu timezone), Gregorian calendar

export DateTime, Duration, add_duration, datetime_diff, monthdays

"""
    DateTime

Represents a datetime with granularity of integer hours (no minutes or seconds).
All datetimes are in UTC (Zulu timezone) and use the Gregorian calendar.
No validation of dates is performed.

# Fields
- `year::Int` - Year
- `month::Int` - Month (1=Jan, 2=Feb, ..., 12=Dec)
- `day::Int` - Day of month (1, 2, 3, ...)
- `hour::Int` - Hour of day (0, 1, 2, ..., 23)
"""
struct DateTime
    year::Int
    month::Int
    day::Int
    hour::Int
end

"""
    Duration

Duration between two datetimes, represented in hours.

# Fields
- `hours::Int` - Duration in hours (can be negative)
"""
struct Duration
    hours::Int
end

# Constructor that takes years, months, days, hours and converts to total hours
function Duration(years::Int, months::Int, days::Int, hours::Int)
    # Convert years and months to days approximately
    # Use standard approximations: 1 year = 365 days, 1 month = 30 days
    total_days = years * 365 + months * 30 + days
    return Duration(total_days * 24 + hours)
end

# Constants for months
const JANUARY = 1
const FEBRUARY = 2
const DECEMBER = 12

"""
    monthdays(month::Int, year::Int) -> Int

Returns the number of days in a given month, accounting for leap years.
"""
function monthdays(month::Int, year::Int)::Int
    MONTHDAYS_NORMAL = [31, -1000, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    if month == FEBRUARY
        # Leap year logic
        if year % 400 == 0
            return 29
        elseif year % 100 == 0
            return 28
        elseif year % 4 == 0
            return 29
        else
            return 28
        end
    else
        return MONTHDAYS_NORMAL[month]
    end
end

"""
    add_duration(dt::DateTime, dur::Duration) -> DateTime

Add a duration to a datetime, returning a new datetime.
Handles month/year boundaries correctly.
"""
function add_duration(dt::DateTime, dur::Duration)::DateTime
    if dur.hours > 0
        hours = dt.hour + dur.hours
        new_hour = mod(hours, 24)

        days = dt.day + div(hours, 24)

        current_month = dt.month
        current_year = dt.year

        # Add days, rolling over months as needed
        while days > monthdays(current_month, current_year)
            days -= monthdays(current_month, current_year)
            current_month += 1
            if current_month == DECEMBER + 1
                current_year += 1
                current_month = JANUARY
            end
        end

        return DateTime(current_year, current_month, days, new_hour)
    else
        hours = dt.hour + dur.hours
        if hours >= 0
            new_hour = mod(hours, 24)
            days = dt.day
        else
            # Need the least non-negative remainder (Euclidean division)
            new_hour = mod(hours, 24)
            days = dt.day + div(hours, 24)
            if new_hour < 0
                new_hour = 24 + new_hour
                days -= 1
            end
        end

        current_month = dt.month
        current_year = dt.year

        # Subtract days, rolling back months as needed
        while days < 1
            current_month -= 1
            if current_month == JANUARY - 1
                current_month = DECEMBER
                current_year -= 1
            end
            days += monthdays(current_month, current_year)
        end

        return DateTime(current_year, current_month, days, new_hour)
    end
end

"""
    hours_to_end_of_year(dt::DateTime) -> Int

Counts the number of hours from the given datetime until the end of the year.
"""
function hours_to_end_of_year(dt::DateTime)::Int
    # Hours to end of current month
    h = (monthdays(dt.month, dt.year) - dt.day) * 24 + (24 - dt.hour)
    current_month = dt.month + 1

    # Add hours for remaining months in the year
    while current_month < DECEMBER + 1
        h += monthdays(current_month, dt.year) * 24
        current_month += 1
    end

    return h
end

"""
    datetime_diff(dt1::DateTime, dt2::DateTime) -> Duration

Compute the difference between two datetimes, returning a Duration.
Result is dt1 - dt2 (positive if dt1 is later than dt2).
"""
function datetime_diff(dt1::DateTime, dt2::DateTime)::Duration
    swap = dt1 < dt2
    if swap
        s = dt1
        f = dt2
    else
        f = dt1
        s = dt2
    end

    hours_diff = 0

    # Add complete years
    while f.year > s.year
        hours_diff += hours_to_end_of_year(s)
        s = DateTime(s.year + 1, JANUARY, 1, 0)
    end

    # Account for remainder in same year
    hours_diff += hours_to_end_of_year(s) - hours_to_end_of_year(f)

    if swap
        hours_diff = -hours_diff
    end

    return Duration(hours_diff)
end

# Operator overloads for convenience
Base.:+(dt::DateTime, dur::Duration) = add_duration(dt, dur)
Base.:-(dt::DateTime, dur::Duration) = add_duration(dt, Duration(-dur.hours))
Base.:-(dt1::DateTime, dt2::DateTime) = datetime_diff(dt1, dt2)

# Comparison operators
Base.:(==)(dt1::DateTime, dt2::DateTime) = (dt1.year == dt2.year &&
                                             dt1.month == dt2.month &&
                                             dt1.day == dt2.day &&
                                             dt1.hour == dt2.hour)

function Base.:(>)(dt1::DateTime, dt2::DateTime)
    if dt1 == dt2
        return false
    end

    if dt1.year > dt2.year
        return true
    elseif dt1.year == dt2.year
        if dt1.month > dt2.month
            return true
        elseif dt1.month == dt2.month
            if dt1.day > dt2.day
                return true
            elseif dt1.day == dt2.day
                return dt1.hour > dt2.hour
            end
        end
    end

    return false
end

Base.:(>=)(dt1::DateTime, dt2::DateTime) = (dt1 > dt2) || (dt1 == dt2)
Base.:(<)(dt1::DateTime, dt2::DateTime) = !(dt1 == dt2) && !(dt1 > dt2)
Base.:(<=)(dt1::DateTime, dt2::DateTime) = (dt1 < dt2) || (dt1 == dt2)

# Duration equality
Base.:(==)(dur1::Duration, dur2::Duration) = dur1.hours == dur2.hours
