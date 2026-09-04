# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os
import omni.client


def list_files_with_extension(folder_path, extension, recursive=False):
    output = []
    result, entries = omni.client.list(folder_path)
    if result != omni.client.Result.OK:
        print(f"Failed to list: {folder_path} (error {result})")
        return

    for entry in entries:
        full_path = os.path.join(folder_path, entry.relative_path)
        if entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN and recursive:
            output.extend(list_files_with_extension(full_path, extension, recursive))
        else:
            if entry.relative_path.endswith(extension):
                output.append(full_path)
    return output
