// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef TYPENAME_GENERATOR_HPP
#define TYPENAME_GENERATOR_HPP

#include "../parser/words.hpp"

#include <string>

std::string typename_to_cuda_code(const TypeName type);

#endif
