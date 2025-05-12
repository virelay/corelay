{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :show-inheritance:
   :members:
   :special-members: __bool__, __call__, __delete__, __delete_attr__, __dir__, __enter__, __exit__, __get__, __get_attr__, __getitem__, __new__, __prepare__, __repr__, __set__, __set_attr__, __set_name__, __setitem__, __tracked__

   {% block attributes %}
   {%- if attributes %}
   .. rubric:: Module Attributes

   .. autosummary::
      :signatures: none

   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block functions %}
   {%- if functions %}
   .. rubric:: Functions

   .. autosummary::
      :signatures: none

   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block classes %}
   {%- if classes %}
   .. rubric:: Classes

   .. autosummary::
      :signatures: none

   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block exceptions %}
   {%- if exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
      :signatures: none

   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

{%- block modules %}
{%- if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:

{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}
