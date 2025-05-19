name : /[a-zA-Z_]([a-zA-Z0-9_])*/
ellipsis : "..."

?axis_name : name | ellipsis

product_axis: axis_name ("*" axis_name)+

?axis_name_or_product: product_axis
    | axis_name

axes : axis_name_or_product ("," axis_name_or_product)+
    | axis_name_or_product

variable : name "[" axes "]"
    | name
    | "[" axes "]"

input_variables : variable ("," variable)*
output_variable : variable?

einsum : input_variables "->" output_variable

?start: einsum

%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS

COMMENT: "//" /[^\n]/*

%ignore COMMENT
%ignore WS
