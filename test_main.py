from main import *
import numpy as np
import pytest
import random

##
## Для корректной работы тестов блок с кодом для ручного ввода в main.py надо закомменировать и использовать константы ниже него
##

SPEED_OF_LIGHT = 299792.458  # Скорость света в км/с
EARTH_R = 6371.0  # Радиус Земли

# Проверка загрузки данных TLE из файла
def test_load_tle_from_file():
    tle_data1 = load_tle_from_file('test_tle_data.txt')
    assert len(tle_data1) > 0
    with pytest.raises(ValueError):
        tle_data2 = load_tle_from_file('test_tle_data_err.txt')

# Проверка преобразования TLE в объект спутника
def test_load_satellite_from_tle():
    tle_data = load_tle_from_file('test_tle_data.txt')
    satellite = load_satellite_from_tle(tle_data[0][0], tle_data[0][1])
    assert satellite is not None

# Проверка функции для расчета позиции спутника
def test_satellite_position():
    tle_data = load_tle_from_file('test_tle_data.txt')
    satellite = load_satellite_from_tle(tle_data[0][0], tle_data[0][1])

    # Тестируем позицию и скорость спутника в различные моменты времени
    position1, velocity1 = satellite_position(satellite, datetime.utcnow())
    assert position1 is not None
    assert velocity1 is not None

    time1 = datetime(2022, 1, 1, 12, 39, 11)
    position2, velocity2 = satellite_position(satellite, time1)
    assert position2 is not None
    assert velocity2 is not None

    time2 = datetime(2025, 7, 8, 1, 0, 0)
    position3, velocity3 = satellite_position(satellite, time2)
    assert position3 is not None
    assert velocity3 is not None

    # Проверяем, что позиции и скорости спутника отличаются
    assert position1 != position2
    assert position1 != position3
    assert velocity1 != velocity2
    assert velocity1 != velocity3

# Проверка функции для проверки видимости спутника
def test_is_satellite_visible():
    tle_data = load_tle_from_file('test_tle_data.txt')
    satellite = load_satellite_from_tle(tle_data[0][0], tle_data[0][1])
    position, velocity = satellite_position(satellite, datetime.utcnow())

    # Тестируем видимость спутника для различных координат наблюдателя
    observer_lat = random.uniform(-90, 90)
    observer_lon = random.uniform(-180, 180)
    visible1, elevation1, distance1 = is_satellite_visible(position, observer_lat, observer_lon)
    observer_lat = random.uniform(-90, 90)
    observer_lon = random.uniform(-180, 180)
    visible2, elevation2, distance2 = is_satellite_visible(position, observer_lat, observer_lon)
    observer_lat = random.uniform(-90, 90)
    observer_lon = random.uniform(-180, 180)
    visible3, elevation3, distance3 = is_satellite_visible(position, observer_lat, observer_lon)

    # Проверяем, что значения видимости, угла возвышения и расстояния не равны None
    assert visible1 is not None
    assert elevation1 is not None
    assert distance1 is not None
    assert visible2 is not None
    assert elevation2 is not None
    assert distance2 is not None
    assert visible3 is not None
    assert elevation3 is not None
    assert distance3 is not None

    # # Проверяем, что значения видимости являются логическими значениями
    assert isinstance(visible1, np.bool_)
    assert isinstance(visible2, np.bool_)
    assert isinstance(visible3, np.bool_)

    # Проверяем, что значения угла возвышения и расстояния типа float
    assert isinstance(elevation1, float)
    assert isinstance(distance1, float)
    assert isinstance(elevation2, float)
    assert isinstance(distance2, float)
    assert isinstance(elevation3, float)
    assert isinstance(distance3, float)

    # Проверяем, что значения угла возвышения и расстояния неотрицательные
    assert distance1 >= 0
    assert distance2 >= 0
    assert distance3 >= 0

    # Проверяем, что значения угла возвышения не превышают 90 градусов
    assert elevation1 <= 90
    assert elevation2 <= 90
    assert elevation3 <= 90

# Проверка функции для расчета прохождений
def test_calculate_passes():
    tle_data = load_tle_from_file('test_tle_data.txt')
    satellite = load_satellite_from_tle(tle_data[0][0], tle_data[0][1])
    observer_lat = random.uniform(-90, 90)
    observer_lon = random.uniform(-180, 180)
    start_time = datetime.utcnow()
    end_time = start_time + timedelta(hours=random.uniform(1, 24))
    passes = calculate_passes(satellite, observer_lat, observer_lon, start_time, end_time)
    assert passes is not None
    if passes != []:
        for pass_data in passes:
            assert len(pass_data) == 4

            # Проверяем, что время начала и конца пролета является объектом datetime
            assert isinstance(pass_data[0], datetime)
            assert isinstance(pass_data[1], datetime)

            # Проверяем, что максимальный угол возвышения и минимальное расстояние являются числами
            assert isinstance(pass_data[2], (int, float))
            assert isinstance(pass_data[3], (int, float))

            # Проверяем, что время начала пролета меньше времени конца пролета
            assert pass_data[0] < pass_data[1]

            # Проверяем, что максимальный угол возвышения положительный
            assert pass_data[2] > 0

            # Проверяем, что минимальное расстояние положительное
            assert pass_data[3] > 0

# Проверка функции для расчета скорости наблюдателя
def test_observer_velocity():
    observer_lat = random.uniform(-90, 90)
    observer_lon = random.uniform(-180, 180)
    velocity = observer_velocity(observer_lat, observer_lon)
    assert velocity is not None
    assert len(velocity) == 3

# Проверка функции для расчета относительной скорости
def test_relative_speed():
    observer_lat = random.uniform(-90, 90)
    observer_lon = random.uniform(-180, 180)
    ob_velocity = observer_velocity(observer_lat, observer_lon)
    satellite_velocity = np.array([1, 2, 3])

    # Проверяем, что аргументы являются массивами NumPy с тремя элементами
    assert isinstance(satellite_velocity, np.ndarray) and satellite_velocity.shape == (3,)
    assert isinstance(ob_velocity, np.ndarray) and ob_velocity.shape == (3,)

    # Проверяем, что элементы массивов являются числами
    assert np.issubdtype(satellite_velocity.dtype, np.number)
    assert np.issubdtype(ob_velocity.dtype, np.number)

    # Проверяем, что элементы массивов не превышают скорость света
    assert np.all(satellite_velocity < SPEED_OF_LIGHT)
    assert np.all(ob_velocity < SPEED_OF_LIGHT)

    speed = relative_speed(satellite_velocity, ob_velocity)
    assert speed is not None

# Проверка функции для расчета доплеровского сдвига
def test_doppler_shift():
    relative_velocity = SPEED_OF_LIGHT / 100
    frequency = 100
    assert isinstance(relative_velocity, (int, float))
    assert relative_velocity < SPEED_OF_LIGHT
    assert isinstance(frequency, (int, float))
    assert frequency > 0
    doppler = doppler_shift(relative_velocity, frequency)
    assert doppler is not None
    expected_doppler_shift = (frequency * relative_velocity) / SPEED_OF_LIGHT
    assert doppler < frequency
    assert doppler == expected_doppler_shift

# Проверка функции для расчета доплеровского сдвига
def test_received_power():
    transmitted_power = 1000
    gain_transmitter = 10
    gain_receiver = 10
    wavelength = 0.1
    distance = 1000

    # Проверяем, что аргументы функции имеют корректные типы и значения
    assert isinstance(transmitted_power, (int, float))
    assert transmitted_power > 0
    assert isinstance(gain_transmitter, (int, float))
    assert isinstance(gain_receiver, (int, float))
    assert isinstance(wavelength, (int, float))
    assert wavelength > 0
    assert isinstance(distance, (int, float))
    assert distance > 0

    # Проверяем полученный результат
    loss = received_power(transmitted_power, gain_transmitter, gain_receiver, wavelength, distance)
    assert loss is not None
    assert isinstance(loss, (int, float))
    assert loss >= 0 and loss < 1
    expected_loss = (transmitted_power * gain_transmitter * gain_receiver * (wavelength ** 2)) / ((4 * np.pi * distance) ** 2)
    assert loss == expected_loss

# Проверка функции для расчета площади покрытия спутника
def test_calculate_coverage_area():
    tle_data = load_tle_from_file('test_tle_data.txt')
    satellite = load_satellite_from_tle(tle_data[0][0], tle_data[0][1])
    position_sat, velocity1 = satellite_position(satellite, datetime.utcnow())
    altitude = 2000

    # Проверяем, что аргументы функции имеют корректные типы и значения
    assert isinstance(position_sat, tuple)
    assert len(position_sat) == 3
    assert isinstance(altitude, (int, float))
    assert altitude > 0

    # Проверяем полученный результат
    coverage_radius = calculate_coverage_area(position_sat, altitude)
    assert coverage_radius is not None
    assert coverage_radius >= 0
    assert isinstance(coverage_radius, (int, float))
    expected_coverage_radius = np.sqrt((altitude + EARTH_R)**2 - EARTH_R**2)
    assert coverage_radius == expected_coverage_radius

# Проверка функции для преобразования широты и долготы в 3D координаты
def test_lat_lon_to_xyz():
    for i in range(10):
        lat = random.uniform(-90, 90)
        lon = random.uniform(-180, 180)
        altitude = random.uniform(0, 2000)

        # Проверяем, что аргументы функции имеют корректные типы и значения
        assert isinstance(lat, (int, float))
        assert isinstance(lon, (int, float))
        assert isinstance(altitude, (int, float))

        # Проверяем полученный результат
        xyz = lat_lon_to_xyz(lat, lon)
        assert isinstance(xyz, tuple)
        assert len(xyz) == 3
        x, y, z = lat_lon_to_xyz(lat, lon, altitude)
        assert x is not None
        assert y is not None
        assert z is not None
        assert isinstance(x, (int, float))
        assert isinstance(y, (int, float))
        assert isinstance(z, (int, float))
        corrected_lon = lon - 180
        R = 6371.0 + altitude
        exp_x = R * np.cos(np.radians(lat)) * np.cos(np.radians(corrected_lon))
        exp_y = R * np.cos(np.radians(lat)) * np.sin(np.radians(corrected_lon))
        exp_z = R * np.sin(np.radians(lat))
        assert x == exp_x
        assert y == exp_y
        assert z == exp_z
