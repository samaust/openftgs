{#- SPDX-License-Identifier: Apache-2.0 -#}
{#- One page per public module: its docstring, summary tables, then every name of its __all__
    (Sphinx reads __all__ because autosummary_ignore_module_all is False). The module attributes
    are every member that is not a function, class or exception, so constants and aliases appear
    whether or not they carry a docstring; automodule leaves them to the autodata entries. The
    rubric and summary blocks follow the layout of Sphinx's own autosummary/module.rst (Sphinx
    9.2.0, BSD-2-Clause, Copyright (c) 2007-2026 by the Sphinx team; source log 303), whose
    license follows, from Sphinx 9.2.0's LICENSE.rst:

    Copyright (c) 2007-2026 by the Sphinx team (see AUTHORS file).
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. -#}
{%- set data = members | reject("in", functions) | reject("in", classes)
    | reject("in", exceptions) | list -%}
{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :members:
   :show-inheritance:
{%- if data %}
   :exclude-members: {{ data | join(", ") }}
{%- endif %}

{% if data %}
   .. rubric:: Module attributes

   .. autosummary::
{% for item in data %}
      {{ item }}
{%- endfor %}
{% endif %}
{% if functions %}
   .. rubric:: Functions

   .. autosummary::
{% for item in functions %}
      {{ item }}
{%- endfor %}
{% endif %}
{% if classes %}
   .. rubric:: Classes

   .. autosummary::
{% for item in classes %}
      {{ item }}
{%- endfor %}
{% endif %}
{% if exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
{% for item in exceptions %}
      {{ item }}
{%- endfor %}
{% endif %}
{% for item in data %}
   .. autodata:: {{ item }}
{% endfor %}
