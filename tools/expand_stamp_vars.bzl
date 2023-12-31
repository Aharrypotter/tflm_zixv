# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----

def expand_stamp_vars(name, template, out):
    """Macro for expanding a template using workspace status variables.

    Typical usage in a BUILD file:

        expand_stamp_vars(
            name = "version",
            template = "_version.py.in",
            out = "_version.py",
        )

    Writes `template` to `out`, expanding references of the form '{KEY}' to the
    value of the corresponding Bazel workspace status variable.
    """

    # This macro uses a genrule to call a helper program at Bazel execution
    # time, because workspace variables are not available until execution time.
    # Workspace variables are generated by bazel on each invocation, and
    # written to the hardcoded files names used below. See the Bazel
    # documentation for the option --workspace_status_command.

    native.genrule(
        name = name,
        srcs = [template],
        outs = [out],
        cmd = "$(location //tools:expand_stamp_vars) " +
              "bazel-out/stable-status.txt " +
              "bazel-out/volatile-status.txt " +
              "<$< >$@",
        tools = [
            "//tools:expand_stamp_vars",
        ],

        # Undocumented, but valid, and the only way to declare the necessary
        # dependencies on {stable,volatile}-status.txt.
        stamp = 1,
    )
