// Default hirn policy: open mode — all agents can perform all actions.
// Replace or extend with restrictive policies for production use.
//
// Example production policies:
//
// // Writers can remember and recall in their realm
// permit(
//     principal in Hirn::Team::"writers",
//     action in [Hirn::Action::"remember", Hirn::Action::"recall", Hirn::Action::"think"],
//     resource in Hirn::Realm::"production"
// );
//
// // Only admins can consolidate or forget
// permit(
//     principal in Hirn::Team::"admins",
//     action in [Hirn::Action::"consolidate", Hirn::Action::"forget", Hirn::Action::"admin"],
//     resource
// );
//
// // Block all access to restricted namespaces unless admin
// forbid(
//     principal,
//     action,
//     resource
// ) when { resource has "classification" && resource.classification == "restricted" }
// unless { principal in Hirn::Team::"admins" };
//
// // Agents with low reputation cannot write
// forbid(
//     principal,
//     action in [Hirn::Action::"remember", Hirn::Action::"connect"],
//     resource
// ) when { principal has "reputation" && principal.reputation < 50 };

// Open mode: permit everything.
permit(principal, action, resource);
