{#- SPDX-License-Identifier: Apache-2.0 -#}
{#- One page per public module: its docstring, summary tables, then every name of its __all__
    (Sphinx reads __all__ because autosummary_ignore_module_all is False). The module attributes
    are every member that is not a function, class or exception, so constants and aliases appear
    whether or not they carry a docstring; automodule leaves them to the autodata entries. The
    rubric and summary blocks follow the layout of Sphinx's own autosummary/module.rst (Sphinx
    9.2.0, BSD-2-Clause, Copyright (c) 2007-2026 by the Sphinx team; source log 303). -#}
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
