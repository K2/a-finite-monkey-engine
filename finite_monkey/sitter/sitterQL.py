"""
TreeSitter query language definitions for Solidity

This module provides TreeSitter query definitions for analyzing Solidity code,
including function detection, taint tracking, and business flow analysis.
"""

def getTagsQuery():
    """
    Default tags query from tree-sitter-solidity
    """
    return """;; Method and Function declarations
(contract_declaration (_
    (function_definition
        name: (identifier) @name) @definition.method))

(source_file
    (function_definition
        name: (identifier) @name) @definition.function)

;; Contract, struct, enum and interface declarations
(contract_declaration
  name: (identifier) @name) @definition.class

(interface_declaration
  name: (identifier) @name) @definition.interface

(library_declaration
  name: (identifier) @name) @definition.interface

(struct_declaration name: (identifier) @name) @definition.class
(enum_declaration name: (identifier) @name) @definition.class
(event_definition name: (identifier) @name) @definition.class

;; Function calls
(call_expression (expression (identifier)) @name ) @reference.call

(call_expression
    (expression (member_expression
        property: (_) @name ))) @reference.call

;; Log emit
(emit_statement name: (_) @name) @reference.class


;; Inheritance

(inheritance_specifier
    ancestor: (user_defined_type (_) @name . )) @reference.class


;; Imports ( note that unknown is not standardised )
(import_directive
  import_name: (_) @name ) @reference.unknown
    """


def flowTrack():
    """
    Query for tracking code flow and function relationships
    """
    return """
        ; Method and Function declarations
(contract_declaration
    (function_definition
        name: (identifier) @name) @definition.method)

(source_file
    (function_definition
        name: (identifier) @name) @definition.function)

;; Contract, struct, enum and interface declarations
(contract_declaration
  name: (identifier) @name) @definition.class

(interface_declaration
  name: (identifier) @name) @definition.interface

(library_declaration
  name: (identifier) @name) @definition.interface

(struct_declaration
  name: (identifier) @name) @definition.class

(enum_declaration
  name: (identifier) @name) @definition.class

(event_definition
  name: (identifier) @name) @definition.event

;; Function calls
(call_expression
    (expression (identifier) @name)) @reference.call

(call_expression
    (expression (member_expression
        object: (identifier) @instantiated_object
        property: (identifier) @method_called))) @reference.call

(call_expression
    (expression (_) @function_name)
    (_ (_) @function_args)) @reference.call

;; Assignment and Taint Flow Tracking
(assignment_expression
    (left (_) @assigned_var)
    (right (_) @source_expr)) @taint.flow

;; Log emit
(emit_statement
    (expression (identifier) @event_name)
    (_ (_) @event_args)) @reference.event

;; Inheritance
(inheritance_specifier
    (user_defined_type (identifier) @name)) @reference.class

;; Imports (fixed for Solidity grammar)
(import_directive
    (string) @name) @reference.import

;; Additional Taint Tracking for Complex Structures
(assignment_expression
    (left (member_expression
        object: (_) @tainted_object
        property: (identifier) @tainted_property))
    (right (_) @source_expr)) @taint.flow

(call_expression
    (expression (_) @function_name)
    (_ (_) @taint_args)) @taint.flow

;; Dangerous Function Tracking
(call_expression
    (expression (identifier) @dangerous_function)
    (_ (_) @dangerous_args))
  (#match? @dangerous_function "selfdestruct|delegatecall|send|call")

;; Constructor Detection
(call_expression
    (expression (identifier) @constructor_call))
  (#match? @constructor_call "^[A-Z].*") @reference.constructor

;; Control Flow Tracking
(if_statement
    (condition (expression) @control_condition)
    (consequence (_) @control_then)
    (alternative (_) @control_else)) @control.flow

;; Cross-Contract Business Flow Tracking
(call_expression
    (expression (member_expression
        object: (identifier) @cross_contract_call
        property: (identifier) @cross_contract_function))) @business.flow

;; Transient Data Storage Tracking
(state_variable_declaration
    (variable_declaration
        name: (identifier) @transient_variable)) @transient.state

(assignment_expression
    (left (identifier) @transient_var)
    (right (_) @transient_value)) @transient.flow

;; Ignore Function Mechanism
(function_definition
    (name (identifier) @ignored_function))
  (#match? @ignored_function "ignore_.*")


        """

def traceWithTaint():
    """
    Specialized query for taint tracking in Solidity
    """
    return """
    ; Method and Function declarations
(contract_declaration (_
    (function_definition
        name: (identifier) @name) @definition.method))

(source_file
    (function_definition
        name: (identifier) @name) @definition.function)

;; Contract, struct, enum and interface declarations
(contract_declaration
  name: (identifier) @name) @definition.class

(interface_declaration
  name: (identifier) @name) @definition.interface

(library_declaration
  name: (identifier) @name) @definition.interface

(struct_declaration name: (identifier) @name) @definition.class
(enum_declaration name: (identifier) @name) @definition.class
(event_definition name: (identifier) @name) @definition.event

;; Function calls
(call_expression (expression (identifier)) @name ) @reference.call

(call_expression
    (expression (member_expression
        (expression (_) @instantiated_object)
        (identifier) @method_called))) @reference.call

(call_expression
    (expression (_) @function_name)
    (_ (_) @function_args)) @reference.call

;; Assignment and Taint Flow Tracking
(assignment_expression
    (expression (_) @assigned_var)
    (expression (_) @source_expr)) @taint.flow

;; Log emit
(emit_statement
    (expression (identifier) @event_name)
    (_ (_) @event_args)) @reference.event

;; Inheritance

(inheritance_specifier
    (user_defined_type (identifier) @name)) @reference.class

;; Imports (fixed for Solidity grammar)
(import_directive
    (string) @name) @reference.import

;; Additional Taint Tracking for Complex Structures
(assignment_expression
    (expression (member_expression
        (expression (_) @tainted_object)
        (identifier) @tainted_property))
    (expression (_) @source_expr)) @taint.flow

(call_expression
    (expression (_) @function_name)
    (_ (_) @taint_args)) @taint.flow

;; Dangerous Function Tracking
(call_expression
    (expression (identifier) @dangerous_function)
    (_ (_) @dangerous_args))
  (#match? @dangerous_function "selfdestruct|delegatecall|send|call")

;; Constructor Detection
(call_expression
    (expression (identifier) @constructor_call)
    (#match? @constructor_call "^[A-Z].*")) @reference.constructor

;; Control Flow Tracking
(if_statement
    (expression) @control_condition
    (statement) @control_then
    (statement)? @control_else) @control.flow

"""

def trace2():
    """
    Alternative trace query with more structured node types
    """
    return """

    ; Correct function declaration with valid node types
(contract_declaration
  body: (contract_body
    (function_definition
      name: (identifier) @name
      (parameter_list)  ; Valid parameter node
      (returns_clause (parameter_list))?  ; Valid returns node
    ) @definition.method
  )
)

; Free function declaration
(source_file
  (function_definition
    name: (identifier) @name
    (parameter_list)
    (returns_clause (parameter_list))?
  ) @definition.function
)

; Method and Function declarations
(contract_declaration
    body: (contract_body
        (function_definition
            (identifier) @name
            (parameters)
            (returns)?)) @definition.method)

(source_file
    (function_definition
        (identifier) @name) @definition.function)

;; Contract, struct, enum and interface declarations
(contract_declaration
  name: (identifier) @name) @definition.class

(interface_declaration
  name: (identifier) @name) @definition.interface

(library_declaration
  name: (identifier) @name) @definition.interface

(struct_declaration
  name: (identifier) @name) @definition.class

(enum_declaration
  name: (identifier) @name) @definition.class

(event_definition
  name: (identifier) @name) @definition.event

;; Function calls
(call_expression
    function: (identifier) @name) @reference.call

(call_expression
    function: (member_expression
        object: (identifier) @instantiated_object
        property: (identifier) @method_called)) @reference.call

;; Assignment and Taint Flow Tracking
(assignment_expression
    left: (_) @assigned_var
    right: (_) @source_expr) @taint.flow

;; Log emit
(emit_statement
    event: (identifier) @event_name
    arguments: (arguments)) @reference.event

;; Inheritance
(inheritance_specifier
    (user_defined_type (identifier) @name)) @reference.class

;; Imports
(import_directive
    (string) @name) @reference.import

;; Additional Taint Tracking
(assignment_expression
    left: (member_expression
        object: (_) @tainted_object
        property: (identifier) @tainted_property)
    right: (_) @source_expr) @taint.flow

;; Dangerous Function Tracking
(call_expression
    function: (identifier) @dangerous_function
    arguments: (arguments))
  (#match? @dangerous_function "selfdestruct|send")

(call_expression
    function: (member_expression
        property: (identifier) @dangerous_function)
    arguments: (arguments))
  (#match? @dangerous_function "delegatecall|call")

;; Constructor Detection
(constructor_definition) @reference.constructor

;; Control Flow
(if_statement
    condition: (_) @control_condition) @control.flow

;; State Variables
(state_variable_declaration
    (variable_declaration (identifier) @state_var)
    (visibility)? @visibility) @state.declaration

;; Mappings
(mapping_type
  key: (_) @mapping_key
  value: (_) @mapping_value) @mapping.definition

"""

# Helper functions for retrieving queries
def getBusinessFlowQuery():
    """Get the business flow tracking query"""
    return flowTrack()

def getVulnerabilityQuery():
    """Get the query for common vulnerability patterns"""
    return traceWithTaint()

def getFunctionsQuery():
    """Get the function query"""
    return """
  ; Method and Function declarations
(contract_declaration (_
    (function_definition
        name: (identifier) @name) @definition.method))

(source_file
    (function_definition
        name: (identifier) @name) @definition.function)

;; Contract, struct, enum and interface declarations
(contract_declaration
  name: (identifier) @name) @definition.class

;; Function calls
(call_expression (expression (identifier)) @name ) @reference.call

(call_expression
    (expression (member_expression
        object: (_) @instantiated_object
        property: (_) @method_called))) @reference.call

(call_expression
    (expression (_) @function_name)
    (_ (_) @function_args)) @reference.call
    
"""